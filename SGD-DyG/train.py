import torch
import torch.nn.functional as F
import torch.optim

from loss import Loss
from models.model import SGDDyG, MultiScaleSGDDyG
from utils import util, DataLoader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.multiscale import build_multiscale_windows, compute_observability_stats


def build_confounder_features(edge_times, time_slices, device):
    edges_per_t = torch.bincount(edge_times, minlength=time_slices).to(torch.float)
    max_edges = torch.clamp(edges_per_t.max(), min=1.0)
    time_norm = edge_times.to(torch.float) / max(time_slices - 1, 1)
    edges_norm = edges_per_t[edge_times] / max_edges
    return torch.stack((time_norm, edges_norm), dim=1).to(device)


def perturb_confounders(confound_features, bucket_ids):
    permuted = confound_features.clone()
    unique_ids = bucket_ids.unique()
    for bucket_id in unique_ids:
        idx = (bucket_ids == bucket_id).nonzero(as_tuple=True)[0]
        if idx.numel() <= 1:
            continue
        shuffled = idx[torch.randperm(idx.numel(), device=confound_features.device)]
        permuted[idx] = permuted[shuffled]
    return permuted


def compute_confounder_loss(
    model,
    views,
    edge_nodes,
    edge_times,
    M,
    stats,
    confound_features,
    alpha,
    gamma,
    sample_rate,
):
    if gamma <= 0:
        return 0.0
    if confound_features is None:
        return 0.0
    if sample_rate <= 0:
        return 0.0

    num_edges = confound_features.size(0)
    sample_size = max(1, int(num_edges * sample_rate))
    sample_idx = torch.randperm(num_edges, device=confound_features.device)[:sample_size]

    edge_nodes_sample = (edge_nodes[0][sample_idx], edge_nodes[1][sample_idx])
    edge_times_sample = edge_times[sample_idx]
    stats_sample = stats[sample_idx]
    confound_sample = confound_features[sample_idx]
    alpha_sample = alpha[sample_idx]

    confound_perturbed = perturb_confounders(confound_sample, edge_times_sample)
    _, _, alpha_perturbed = model(
        views, edge_nodes_sample, edge_times_sample, M, stats_sample, confound_perturbed, False
    )
    return gamma * F.l1_loss(alpha_sample, alpha_perturbed)


def train(args, num_feature, lr, lam, tau):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    TS, labels, A_train, A_val, A_test, N = DataLoader.load_data(args, device)
    train_adj = A_train[:-1]
    val_adj = A_val[:-1]
    test_adj = A_test[:-1]

    T = len(train_adj)
    M = util.func_createM(T, args.bandwidth, args.m_choice, device)

    edges_train, target_train, edges_val, target_val, K_val, edges_test, target_test, K_test = util.split_data(labels, TS)
    train_edge_nodes, val_edge_nodes, test_edge_nodes = util.get_all_edges_nodes(edges_train, edges_val, edges_test, N)

    enable_cl = args.enable_cl

    for run in range(args.num_runs):
        util.set_seed(args.seed)

        save_res_fname, save_model_folder = util.get_save_parameter(lr, lam, num_feature, run, tau, args)
        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder, model_name=args.model_name)

        args_summary = util.stringify_args(args)

        base_encoder = SGDDyG(T, N,
                              hidden_features=args.hidden_features,
                              num_feature=num_feature,
                              out_features=1,
                              bandwidth=args.bandwidth,
                              tgc_dropout=args.tgc_dropout,
                              fft_dropout=args.fft_dropout,
                              fft=args.fft,
                              tensor_con=args.tensor_con)

        if args.multi_scale:
            confounder_features = 2 if args.enable_deconfound else 0
            model = MultiScaleSGDDyG(base_encoder, selector_hidden=args.selector_hidden_dim, time_slices=T, num_scales=3,
                                     prior_beta=args.prior_beta, confounder_features=confounder_features,
                                     confounder_hidden=args.confounder_hidden_dim,
                                     deconfound_eta=args.deconfound_eta,
                                     deconfound_alpha_coeff=args.deconfound_alpha_coeff)
            train_views = build_multiscale_windows(train_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)
            val_views = build_multiscale_windows(val_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)
            test_views = build_multiscale_windows(test_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)

            train_stats = compute_observability_stats(edges_train, train_adj, args.bandwidth, N)
            val_stats = compute_observability_stats(edges_val, val_adj, args.bandwidth, N)
            test_stats = compute_observability_stats(edges_test, test_adj, args.bandwidth, N)
            train_edge_times = edges_train[0]
            val_edge_times = edges_val[0]
            test_edge_times = edges_test[0]
            train_confound = build_confounder_features(train_edge_times, T, device)
            val_confound = build_confounder_features(val_edge_times, T, device)
            test_confound = build_confounder_features(test_edge_times, T, device)
        else:
            model = base_encoder
            train_views, val_views, test_views = train_adj, val_adj, test_adj
            train_stats = val_stats = test_stats = None
            train_edge_times = val_edge_times = test_edge_times = None
            train_confound = val_confound = test_confound = None

        model = model.to(device=device)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        criterion = Loss(base_encoder.X, lam, enable_cl, tau,
                         sharpness_coeff=args.sharpness_coeff if args.multi_scale else 0.0)

        logs = [f"Args: {args_summary}"]
        for ep in range(1, args.epochs + 1):
            optimizer.zero_grad()

            model.train()
            if enable_cl:
                if args.multi_scale:
                    output_train, h1, alpha_train = model(
                        train_views, train_edge_nodes, train_edge_times, M, train_stats, train_confound, False
                    )
                    _, h2, _ = model(train_views, train_edge_nodes, train_edge_times, M, train_stats, train_confound, True)
                    loss_train = criterion(output_train, target_train, h1, h2, alpha_train)
                else:
                    output_train, h1 = model(train_views, train_edge_nodes, M, False)
                    _, h2 = model(train_views, train_edge_nodes, M, True)
                    loss_train = criterion(output_train, target_train, h1, h2)
            else:
                if args.multi_scale:
                    output_train, h1, alpha_train = model(
                        train_views, train_edge_nodes, train_edge_times, M, train_stats, train_confound, False
                    )
                    loss_train = criterion(output_train, target_train, h1, alpha=alpha_train)
                else:
                    output_train, _ = model(train_views, train_edge_nodes, M, False)
                    loss_train = criterion(output_train, target_train)
            train_metrics = util.compute_metrics(output_train, target_train, edges_train)

            if args.multi_scale and args.enable_deconfound:
                loss_train = loss_train + compute_confounder_loss(
                    model,
                    train_views,
                    train_edge_nodes,
                    train_edge_times,
                    M,
                    train_stats,
                    train_confound,
                    alpha_train,
                    args.deconfound_gamma,
                    args.deconfound_sample_rate,
                )

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()

                if enable_cl:
                    if args.multi_scale:
                        output_val, h1, alpha_val = model(
                            val_views, val_edge_nodes, val_edge_times, M, val_stats, val_confound, False
                        )
                        _, h2, _ = model(val_views, val_edge_nodes, val_edge_times, M, val_stats, val_confound, True)
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, h2, alpha_val[-K_val:])
                    else:
                        output_val, h1 = model(val_views, val_edge_nodes, M, False)
                        _, h2 = model(val_views, val_edge_nodes, M, True)
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, h2)
                else:
                    if args.multi_scale:
                        output_val, h1, alpha_val = model(
                            val_views, val_edge_nodes, val_edge_times, M, val_stats, val_confound, False
                        )
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, alpha=alpha_val[-K_val:])
                    else:
                        output_val, _ = model(val_views, val_edge_nodes, M, False)
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:])

                val_metrics = util.compute_metrics(output_val[-K_val:], target_val[-K_val:], edges_val[:, -K_val:])

            log = util.log_metric('Train', ep, train_metrics, loss_train)
            log += util.log_metric('Val', ep, val_metrics, loss_val)
            print(log)
            logs.append(log)

            val_metric_indicator = []
            for metric_name in val_metrics.keys():
                val_metric_indicator.append((metric_name, val_metrics[metric_name], True))
            early_stop = early_stopping.step(val_metric_indicator, model)
            if early_stop:
                break

        metric_names = ['AP', 'ROC_AUC']
        for metric_name in metric_names:
            early_stopping.load_checkpoint(model, metric_name)
            model.eval()
            if args.multi_scale:
                output_test, _, _ = model(test_views, test_edge_nodes, test_edge_times, M, test_stats, test_confound, False)
            else:
                output_test, _ = model(test_views, test_edge_nodes, M, False)

            test_metric = util.compute_metrics(output_test[-K_test:], target_test[-K_test:], edges_test[:, -K_test:])
            log = (f"Test {metric_name}: {test_metric.get(metric_name)} in Val "
                   f"{metric_name}: {early_stopping.best_metrics.get(metric_name)}")
            print(log)
            logs.append(log)

        with open(save_res_fname, 'w') as f:
            f.write("\n".join(logs))
        print("Results saved for single trial")


def main():
    args = get_link_prediction_args()
    train(args, args.num_feature, args.lr, args.lam, args.tau)


if __name__ == '__main__':
    main()
