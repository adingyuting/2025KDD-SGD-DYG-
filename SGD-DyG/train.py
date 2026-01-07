import torch.optim

from loss import Loss
from models.model import SGDDyG, MultiScaleSGDDyG
from utils import util, DataLoader
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.multiscale import build_multiscale_windows, compute_observability_stats


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
            model = MultiScaleSGDDyG(base_encoder, selector_hidden=args.selector_hidden_dim, num_scales=3)
            train_views = build_multiscale_windows(train_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)
            val_views = build_multiscale_windows(val_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)
            test_views = build_multiscale_windows(test_adj, args.bandwidth, args.decay_lambda, args.persistence_threshold)

            train_stats = compute_observability_stats(edges_train, train_adj, args.bandwidth, N)
            val_stats = compute_observability_stats(edges_val, val_adj, args.bandwidth, N)
            test_stats = compute_observability_stats(edges_test, test_adj, args.bandwidth, N)
        else:
            model = base_encoder
            train_views, val_views, test_views = train_adj, val_adj, test_adj
            train_stats = val_stats = test_stats = None

        model = model.to(device=device)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
        criterion = Loss(base_encoder.X, lam, enable_cl, tau,
                         sharpness_coeff=args.sharpness_coeff if args.multi_scale else 0.0)

        logs = []
        for ep in range(1, args.epochs + 1):
            optimizer.zero_grad()

            model.train()
            if enable_cl:
                if args.multi_scale:
                    output_train, h1, alpha_train = model(train_views, train_edge_nodes, M, train_stats, False)
                    _, h2, _ = model(train_views, train_edge_nodes, M, train_stats, True)
                    loss_train = criterion(output_train, target_train, h1, h2, alpha_train)
                else:
                    output_train, h1 = model(train_views, train_edge_nodes, M, False)
                    _, h2 = model(train_views, train_edge_nodes, M, True)
                    loss_train = criterion(output_train, target_train, h1, h2)
            else:
                if args.multi_scale:
                    output_train, h1, alpha_train = model(train_views, train_edge_nodes, M, train_stats, False)
                    loss_train = criterion(output_train, target_train, h1, alpha=alpha_train)
                else:
                    output_train, _ = model(train_views, train_edge_nodes, M, False)
                    loss_train = criterion(output_train, target_train)
            train_metrics = util.compute_metrics(output_train, target_train, edges_train)

            loss_train.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()

                if enable_cl:
                    if args.multi_scale:
                        output_val, h1, alpha_val = model(val_views, val_edge_nodes, M, val_stats, False)
                        _, h2, _ = model(val_views, val_edge_nodes, M, val_stats, True)
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, h2, alpha_val[-K_val:])
                    else:
                        output_val, h1 = model(val_views, val_edge_nodes, M, False)
                        _, h2 = model(val_views, val_edge_nodes, M, True)
                        loss_val = criterion(output_val[-K_val:], target_val[-K_val:], h1, h2)
                else:
                    if args.multi_scale:
                        output_val, h1, alpha_val = model(val_views, val_edge_nodes, M, val_stats, False)
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
                output_test, _, _ = model(test_views, test_edge_nodes, M, test_stats, False)
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
