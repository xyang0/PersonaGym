#!/bin/bash

MODEL_CARD=gpt-4o-mini-2024-07-18
BENCHMARK=benchmark-v1

python run.py \
    --benchmark ${BENCHMARK} \
    --model ${MODEL_CARD} \
    --model_name ${MODEL_CARD}_${BENCHMARK} \
    --save_name ${MODEL_CARD}_${BENCHMARK}_PScore \
    --eval_1_only

python average_score.py \
    --save_name ${MODEL_CARD}_${BENCHMARK}_PScore
