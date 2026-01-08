#!/usr/bin/env bash
set -e

DATASET="wiki_gl"
RUNS=1
EPOCHS=100

NUM_FEATURES=(8)
LAYERS=(1)
HIDDEN_FEATURES=(16)

LR_LIST=(0.005 0.01 0.02)
WEIGHT_DECAY_LIST=(0.0005 0.001)
LAM_LIST=(0.03 0.05 0.07)
TAU_LIST=(0.1 0.3 0.5)

BANDWIDTH_LIST=(20)
M_CHOICE_LIST=(2)
FFT_LIST=(True)
TENSOR_CON_LIST=(True)
ENABLE_CL_LIST=(True)
TGC_DROPOUT_LIST=(0.75)
FFT_DROPOUT_LIST=(0.75)

MULTI_SCALE_LIST=(True)
DECAY_LAMBDAS=(0.8)
PERSIST_THRESHOLDS=(2)
SELECTOR_HIDDEN_LIST=(64)
SHARPNESS_LIST=(1e-3)
PRIOR_BETA_LIST=(1.0)

for NUM_FEATURE in "${NUM_FEATURES[@]}"; do
  for LAYER in "${LAYERS[@]}"; do
    for HIDDEN_FEATURE in "${HIDDEN_FEATURES[@]}"; do
      for LR in "${LR_LIST[@]}"; do
        for WEIGHT_DECAY in "${WEIGHT_DECAY_LIST[@]}"; do
          for LAM in "${LAM_LIST[@]}"; do
            for TAU in "${TAU_LIST[@]}"; do
              for BANDWIDTH in "${BANDWIDTH_LIST[@]}"; do
                for M_CHOICE in "${M_CHOICE_LIST[@]}"; do
                  for FFT in "${FFT_LIST[@]}"; do
                    for TENSOR_CON in "${TENSOR_CON_LIST[@]}"; do
                      for ENABLE_CL in "${ENABLE_CL_LIST[@]}"; do
                        for TGC_DROPOUT in "${TGC_DROPOUT_LIST[@]}"; do
                          for FFT_DROPOUT in "${FFT_DROPOUT_LIST[@]}"; do
                            for MULTI_SCALE in "${MULTI_SCALE_LIST[@]}"; do
                              for DECAY_LAMBDA in "${DECAY_LAMBDAS[@]}"; do
                                for PERSIST_THRESHOLD in "${PERSIST_THRESHOLDS[@]}"; do
                                  for SELECTOR_HIDDEN in "${SELECTOR_HIDDEN_LIST[@]}"; do
                                    for SHARPNESS in "${SHARPNESS_LIST[@]}"; do
                                      for PRIOR_BETA in "${PRIOR_BETA_LIST[@]}"; do
                                      python SGD-DyG/train.py \
                                        --dataset_name "$DATASET" \
                                        --num_runs "$RUNS" \
                                        --epochs "$EPOCHS" \
                                        --num_feature "$NUM_FEATURE" \
                                        --layer "$LAYER" \
                                        --hidden_feature "$HIDDEN_FEATURE" \
                                        --lr "$LR" \
                                        --weight_decay "$WEIGHT_DECAY" \
                                        --lam "$LAM" \
                                        --tau "$TAU" \
                                        --bandwidth "$BANDWIDTH" \
                                        --m_choice "$M_CHOICE" \
                                        --fft "$FFT" \
                                        --tensor_con "$TENSOR_CON" \
                                        --enable_cl "$ENABLE_CL" \
                                        --tgc_dropout "$TGC_DROPOUT" \
                                        --fft_dropout "$FFT_DROPOUT" \
                                        --multi_scale "$MULTI_SCALE" \
                                        --decay_lambda "$DECAY_LAMBDA" \
                                        --persistence_threshold "$PERSIST_THRESHOLD" \
                                        --selector_hidden_dim "$SELECTOR_HIDDEN" \
                                        --sharpness_coeff "$SHARPNESS" \
                                        --prior_beta "$PRIOR_BETA"
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
