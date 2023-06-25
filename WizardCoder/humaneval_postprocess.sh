#!/usr/bin/env bash
set -x

temp=0.2
pred_num=1

output_path=preds/starcoderbase-code_evol-Greedy_Decode/T${temp}_N${pred_num}

echo 'Output path: '$output_path
python src/process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

evaluate_functional_correctness ${output_path}.jsonl