#!/usr/bin/env bash
set -x

python3 src/inference_wizardcoder.py \
    --base_model "/dataset/home/liaoxingyu/Llama-X/output/alpaca_evol_instruct_v1-starcoderplus-15b-fp16-zero_dp-plr2e-5-mlr0-mbsz16-gbsz512-ctxlen2048-tokn70k_piece-ep3-wmup30/checkpoint-800" \
    --input_data_path "data.jsonl" \
    --output_data_path "result.jsonl"


# --base_model "/dataset/home/liaoxingyu/models/WizardCoder-15B-V1.0" \
# --base_model "/dataset/home/liaoxingyu/models/starcoderplus" \