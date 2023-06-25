from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob
import re
import argparse

parser = argparse.ArgumentParser()



# Inputs
parser.add_argument(
    '--path',
    type=str,
    help="")
parser.add_argument(
    '--out_path',
    type=str,
    help="")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='')

args = parser.parse_args()

def truncate(completion, truncate_before_pattern):
    def find_re(string, pattern, start_pos):
        m = pattern.search(string, start_pos)
        return m.start() if m else -1

    terminals = [re.compile(pattern, re.MULTILINE) for pattern in truncate_before_pattern]

    prints = list(re.finditer("^print", completion, re.MULTILINE))

    if prints:
        completion = completion[: prints[0].start()]

    # defs = list(re.finditer("^def", completion, re.MULTILINE))

    # if len(defs) > 1:
    #     print(completion)
    #     completion = completion[: defs[1].start()]

    start_pos = 0

    terminals_pos = [
        pos for pos in [find_re(completion, terminal, start_pos) for terminal in terminals] if pos != -1
    ]

    return completion[: min(terminals_pos)] if terminals_pos else completion


files = sorted(glob.glob(f'{args.path}/*.jsonl'))
print(f"{len(files)} files in {args.path}")

problems = read_problems()

output = []
a = 0
for code_file in files:
    codes = list(stream_jsonl(code_file))
    if args.add_prompt:

        for code in codes:
            task_id = code['task_id']
            prompt = problems[task_id]['prompt']
            completion = code['completion']
            completion = completion.replace("\r", "")
            if '```python' in completion:
                def_line = completion.index('```python')
                completion = completion[def_line:].strip()
                completion = completion.replace('```python', '')
                try:
                    next_line = completion.index('```')
                    completion = completion[:next_line].strip()
                except Exception:
                    a += 1
                    print(completion)
                    print("================\n")
            completion = truncate(completion, ["^#", re.escape("<|endoftext|>"), "^'''"])

            code['completion'] = completion

    output += codes

print(f"save to {args.out_path}")
write_jsonl(args.out_path, output)
print(a)