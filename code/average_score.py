# Use it to average scores from all personas

import json
import argparse
import os
import numpy as np

from personas import benchmark_personas


def arg_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, help="flag for input data", default="gpt-4o_benchmark-v1_PScore")
    args = parser.parse_args()
    return args


def main(args):
    persona_list = benchmark_personas
    score_dir = f"../scores/{args.save_name}"
    over_score_dict = {}
    for persona in persona_list:
        score_file_path = f"{score_dir}/{persona}_score.json"
        if os.path.exists(score_file_path):
            with open(score_file_path, 'r') as file:
                score_dict = json.load(file)
                for key, value in score_dict.items():
                    if key in over_score_dict:
                        over_score_dict[key].append(value)
                    else:
                        over_score_dict[key] = [value]

    for key, value in over_score_dict.items():
        aver_score = np.average(value)
        score_std = np.std(value)
        print(f"{key} ({len(value)}/{len(persona_list)}): {aver_score} ({score_std})")


if __name__ == "__main__":
    args = arg_loader()
    main(args)
