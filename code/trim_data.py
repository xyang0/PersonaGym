# Use it to shrink down the size of testing data but keep the persona and task coverage.
# E.g., generate benchmark-v1_personagym-light (i.e., PersonaGym-Light) with "python trim_data.py --trimmed benchmark-v1_personagym-light"

import json
import argparse
import os

from eval_tasks import tasks
from personas import benchmark_personas


def arg_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=str, help="flag for input data", default="benchmark-v1")
    parser.add_argument("--trimmed", type=str, help="flag for trimmed data", default="benchmark-v1_trimmed")
    parser.add_argument('--num_sample', type=int, help="number of samples in each task to keep from top", default=1)
    parser.add_argument('--trim_responses', help="flag for trimming saved answers instead of benchmark questions", action='store_true')
    args = parser.parse_args()
    return args


def trim_data(persona, original, trimmed, num_sample=1, data_type="questions"):
    output_file_suffix = "_qa.json" if data_type == "results" else ".json"

    orig_dir = f"../{data_type}/{original}"
    if not os.path.exists(orig_dir):
        print(f"No original data directory {dir}")
        exit(0)

    input_file_path = f'{orig_dir}/{persona}{output_file_suffix}'
    if not os.path.exists(input_file_path):
        print(f"No JSON file {input_file_path}")
        exit(0)

    with open(input_file_path, 'r') as file:
        samples = json.load(file)

    trimmed_samples = {}
    for task in tasks:
        trimmed_samples[task] = samples[task][:num_sample]

    trim_dir = f"../{data_type}/{trimmed}"
    if not os.path.exists(trim_dir):
        os.makedirs(trim_dir)

    with open(f'{trim_dir}/{persona}{output_file_suffix}', 'w') as file:
        json.dump(trimmed_samples, file, indent=4)


def main(args):
    persona_list = benchmark_personas
    for persona in persona_list:
        data_type = "results" if args.trim_responses else "questions"
        trim_data(persona, args.original, args.trimmed, args.num_sample, data_type)


if __name__ == "__main__":
    args = arg_loader()
    main(args)
