#!/usr/bin/env bash
set -x

model="/dataset/home/liaoxingyu/Llama-X/output/code_evol_v1-starcoderplus-15b-fp16-zero_dp-plr2e-5-mlr0-mbsz16-gbsz256-ctxlen2048-tokn20k_piece-ep3-wmup30/checkpoint-225"
temp=0.2
max_len=2048
pred_num=1
num_seqs_per_iter=1

output_path=preds/starcoderplus-code_evol-Greedy_Decode/T${temp}_N${pred_num}

mkdir -p ${output_path}
echo 'Output path: '$output_path
echo 'Model to eval: '$model

# 164 problems, 21 per GPU if GPU=8
index=0
gpu_num=8
for ((i = 0; i < $gpu_num; i++)); do
  start_index=$((i * 21))
  end_index=$(((i + 1) * 21))

  gpu=$((i))
  echo 'Running process #' ${i} 'from' $start_index 'to' $end_index 'on GPU' ${gpu}
  ((index++))
  (
    CUDA_VISIBLE_DEVICES=$gpu python3 src/humaneval_gen.py --model ${model} \
      --start_index ${start_index} --end_index ${end_index} --temperature ${temp} \
      --num_seqs_per_iter ${num_seqs_per_iter} --N ${pred_num} --max_len ${max_len} --output_path ${output_path} --g
  ) &
  if (($index % $gpu_num == 0)); then wait; fi
done

python src/process_humaneval.py --path ${output_path} --out_path ${output_path}.jsonl --add_prompt

evaluate_functional_correctness ${output_path}.jsonl
