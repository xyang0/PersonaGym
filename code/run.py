from utils import run_model
from eval_tasks import *
import ast
import argparse
import os
import json
from personas import benchmark_personas
import logging
import re

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GPT_4O = "gpt-4o-2024-08-06"
SETTINGS_MODEL = GPT_4O
QUESTION_MODEL = GPT_4O
EXAMPLE_MODEL = GPT_4O
EVAL_1 = GPT_4O
EVAL_2 = "meta-llama/Llama-3-70b-chat-hf"


def extract_list(original_string):
    list_string = original_string.replace("```python", "")
    list_string = list_string.replace("```", "")
    list_string = list_string.lstrip().rstrip()
    actual_list = ast.literal_eval(list_string)
    return actual_list


# Short-listing relevant scenarios/enviornments
def select_settings(persona):
    settings_prompt = f'''
                        Given the following persona description, select the most relevant settings from the given settings options for the persona. Your output must only be the selected settings in a python list format with no other verbose.
                        Persona: {persona}
                        Settings: {settings_list}
                        Selected Settings:
                      '''
    selected_settings  = run_model(input_prompt=settings_prompt, model_card=SETTINGS_MODEL)
    selected_settings = extract_list(selected_settings)
    return selected_settings


# Generate relevant questions given scenarios
def gen_questions(persona, settings, num_questions=10, addr="127.0.0.1:8000"):
    questions = {task:[] for task in tasks}

    for task in tasks:
        description = question_requirements[task]
        question_prompt = f'''
                            You are tasked with determining if a person with the given persona description is able to answer questions related to {settings} that specifically test the given evaluation task. Generate exactly {num_questions} challenging multi-step questions to do this where the questions are intended to be asked directly to the persona. You may use the question description below to guide you. Your output must be the generated questions in a python list format with no other verbose.
                            Persona: {persona}
                            Settings: {settings}
                            Evaluation Task: {task}
                            Questions Description: {description}
                            Questions: 
                      '''
        for _ in range(5):
            try:
                task_questions  = run_model(input_prompt=question_prompt, model_card=QUESTION_MODEL, addr=addr)
                task_questions = extract_list(task_questions)
            except Exception as e:
                continue
            if len(task_questions) == num_questions:
                break

        questions[task].extend(task_questions)

    return questions


def process_examples(text):
    matches = re.findall(r'Score (\d+): *Response - *"?(.*?)"?(?=\n*Score \d+: *Response -|$)', text, re.S)
    processed_text = '\n\n'.join(f'Score {score}: \"{response.strip()}\"' for score, response in matches)

    lines = processed_text.split("\n")
    filtered_lines = [line for line in lines if line.startswith("Score")]

    return "\n\n".join(filtered_lines)


def parse_full_examples(text):
    rubrics = re.split(r'Rubric \d+ Examples:', text)
    if rubrics[0].strip() == '':
        rubrics.pop(0)
    rubrics = [rubric.strip() for rubric in rubrics]
    
    return rubrics


def gen_score_examples(persona, qa, rubric, model):
    examples_rubric = open(f'../prompts/score_examples/parallel_examples.txt').read()
    rubrics = []
    for question, _ in qa:
        score_prompt = open(f'../prompts/score_examples/prompt.txt').read()
        score_prompt = score_prompt.format(persona = persona, question = question, rubric = rubric)
        rubrics.append(score_prompt)

    prompt = examples_rubric.format(rubrics=rubrics)
    examples = run_model(input_prompt=prompt, temperature=0, top_p=0, model_card=model)
    examples = process_examples(examples)

    return examples


## This function does not use parallel examples -> Instead, generate examples one by one. (Original PersonaGym code has bugs to split multiple examples if using parallel.)
def gen_score_examples_non_parallel(persona, qa, rubric, model):
    examples_rubric = open(f'../prompts/score_examples/examples.txt').read()
    examples = []
    for question, _ in qa:
        score_prompt = open(f'../prompts/score_examples/prompt.txt').read()
        score_prompt = score_prompt.format(persona = persona, question = question, rubric = rubric)
        rubrics = [score_prompt]

        prompt = examples_rubric.format(rubrics=rubrics)
        example = run_model(input_prompt=prompt, temperature=0, top_p=0, model_card=model)
        example = process_examples(example)
        examples.append(example)

    return examples


def parse_rubric(rubric):
    match_segment = re.search(r"Therefore, the final score is\s*(\d+)", rubric)
    if match_segment:
        return int(match_segment.group(1))
    return 0


def format_rubrics(persona, rubric, qa):
    sys_prompt = open(f'../prompts/rubric_grading/sys_prompt.txt').read()
    prompt_outline = open(f'../prompts/rubric_grading/prompt.txt').read()
    rubrics = []

    # examples = gen_score_examples(persona, qa, rubric, EXAMPLE_MODEL)  ## Notice: not using parallel example
    examples = gen_score_examples_non_parallel(persona, qa, rubric, EXAMPLE_MODEL)
    for i in range(len(qa)):
        question, answer = qa[i]
        score_examples = examples[i]
        formatted_rubric = rubric.format(persona = persona, question = question, response = answer, score_example = score_examples)
        rubrics.append(formatted_rubric)

    scoring_prompt = prompt_outline.format(rubrics = rubrics)
    # print(f"-*-\nProcessed scoring prompt:\n{scoring_prompt}\n----****----")

    return sys_prompt, scoring_prompt


def parse_evaluations(text):
    pattern = r'\(\d+\) Evaluation:(.*?)(?=\(\d+\) Evaluation:|$)'
    evaluations = re.findall(pattern, text, re.DOTALL)
    evaluations = [eval.strip() for eval in evaluations]
    return evaluations


def calculate_modified_average(score_list):
    total_sum = sum(score_list)
    zero_count = score_list.count(0)
    mod_total = len(score_list) - zero_count

    return total_sum / mod_total if mod_total > 0 else total_sum


def score_rubrics(sys_prompt, scoring_prompt, num_evals=1, addr="127.0.0.1:8000"):
    scores = []

    for _ in range(num_evals):
        evaluator1 = run_model(input_prompt=scoring_prompt, temperature=0, top_p=0, model_card=EVAL_1, system=sys_prompt, addr=addr)
        evaluator2 = run_model(input_prompt=scoring_prompt, temperature=0, top_p=0, model_card=EVAL_2, system=sys_prompt, addr=addr)

        evaluator1 = parse_evaluations(evaluator1)
        evaluator2 = parse_evaluations(evaluator2)

        scores1 = [parse_rubric(rubric) for rubric in evaluator1]
        scores2 = [parse_rubric(rubric) for rubric in evaluator2]

        score1 = calculate_modified_average(scores1)
        score2 = calculate_modified_average(scores2)

        scores.append(score1)
        scores.append(score2)

    return sum(scores) / len(scores)


def score_rubrics_single_model(sys_prompt, scoring_prompt, model_card=EVAL_1, num_evals=1, addr="127.0.0.1:8000"):
    scores = []

    for _ in range(num_evals):
        evaluator1 = run_model(input_prompt=scoring_prompt, temperature=0, top_p=0, model_card=model_card, system=sys_prompt, addr=addr)
        evaluator1 = parse_evaluations(evaluator1)
        scores1 = [parse_rubric(rubric) for rubric in evaluator1]
        score1 = calculate_modified_average(scores1)
        scores.append(score1)

    return sum(scores) / len(scores)


def gen_answers(persona, questions, model, addr="127.0.0.1:8000"):
    task_to_qa = {}

    for task in questions:
        task_to_qa[task] = []
        task_questions = questions[task]

        for question in task_questions:
            answer = run_model(input_prompt=question, persona=persona, model_card=model, addr=addr)
            task_to_qa[task].append((question, answer))
    
    return task_to_qa


def score_answers(persona, task_to_qa, score_example=True, addr="127.0.0.1:8000"):
    scores = {task:[] for task in task_to_qa}
    for task in task_to_qa:
        for i in range(0, len(task_to_qa[task]), 5):
            selected_qa = task_to_qa[task][i: i + 5]
            rubric = open(f'../rubrics/{task}.txt').read()
            sys_prompt, scoring_prompt = format_rubrics(persona, rubric, selected_qa)

            scores[task].append(score_rubrics(sys_prompt, scoring_prompt, addr=addr))

    for task in scores:
        scores[task] = sum(scores[task]) / len(scores[task])

    return scores


def score_answers_single_model(persona, task_to_qa, model_card=EVAL_1, score_example=True, addr="127.0.0.1:8000"):
    scores = {task:[] for task in task_to_qa}
    for task in task_to_qa:
        for i in range(0, len(task_to_qa[task]), 5):
            selected_qa = task_to_qa[task][i: i + 5]
            rubric = open(f'../rubrics/{task}.txt').read()
            sys_prompt, scoring_prompt = format_rubrics(persona, rubric, selected_qa)
            logger.info(f"Sys Prompt: {sys_prompt}\n\nScoring Prompt: {scoring_prompt}")
            scores[task].append(score_rubrics_single_model(sys_prompt, scoring_prompt, model_card=model_card, addr=addr))

    for task in scores:
        scores[task] = sum(scores[task]) / len(scores[task])

    return scores


def save_responses(persona, task_to_qa, model_name):
    dir = f"../results/{model_name}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f'{dir}/{persona}_qa.json', 'w') as file:
        json.dump(task_to_qa, file, indent=4)


def save_scores(save_name, scores, persona=""):
    dir = f"../scores/{save_name}"
    if not os.path.exists(dir):
        os.makedirs(dir)

    filename = f'{dir}/{persona}_score.json' if persona else f'{dir}/scores.json'
    with open(filename, 'w') as file:
        json.dump(scores, file, indent=4)


def load_questions(persona, saved_questions):
    dir = f"../questions/{saved_questions}"
    if not os.path.exists(dir):
        print(f"No questions directory {dir}")
        exit(0)
    
    file_path = f'{dir}/{persona}.json'
    if not os.path.exists(file_path):
        print(f"No JSON file {file_path}")
        exit(0)

    with open(file_path, 'r') as file:
        questions = json.load(file)

    return questions


def load_responses(persona, saved_responses): 
    dir = saved_responses
    if not os.path.exists(dir):
        print(f"No responses directory {saved_responses}")
        exit(0)
    
    file_path = f'{dir}/{persona}_qa.json'
    if not os.path.exists(file_path):
        print(f"No JSON file {file_path}")
        exit(0)

    with open(file_path, 'r') as file:
        task_to_qa = json.load(file)

    return task_to_qa


def main(persona, model, model_name=None, saved_questions=None, saved_responses=None, eval_1_only=False, eval_2_only=False, addr="127.0.0.1:8000"):
    if saved_responses:
        task_to_qa = load_responses(persona, saved_responses)

    else:
        if saved_questions:
            questions = load_questions(persona, saved_questions)

        else:
            settings = select_settings(persona)
            questions = gen_questions(persona, settings, addr=addr)

        task_to_qa = gen_answers(persona, questions, model, addr=addr)

    if eval_1_only:
        scores = score_answers_single_model(persona, task_to_qa, model_card=EVAL_1, addr=addr)
    elif eval_2_only:
        scores = score_answers_single_model(persona, task_to_qa, model_card=EVAL_2, addr=addr)
    else:
        scores = score_answers(persona, task_to_qa, addr=addr)

    overall = 0
    for task in scores:
        overall += scores[task]

    overall /= len(scores.keys())
    scores["PersonaScore"] = overall

    if model_name:
        save_responses(persona, task_to_qa, model_name)

    return scores


def main_get_benchmark_answers(persona, model, model_name=None, saved_questions=None, saved_responses=None, addr="127.0.0.1:8000"):
    if saved_responses:
        return
    else:
        if saved_questions:
            questions = load_questions(persona, saved_questions)
        else:
            return

        task_to_qa = gen_answers(persona, questions, model, addr=addr)

    if model_name:
        save_responses(persona, task_to_qa, model_name)


def arg_loader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona_list", type=str, help="List of personas", default="[]")
    parser.add_argument("--model", type=str, help="A valid model name from the api options of: OpenAI, Claude, TogetherAI", default="meta-llama/Llama-2-70b-chat-hf")
    parser.add_argument("--model_name", help="Model name to save results", default=None)
    parser.add_argument("--saved_questions", help="Path to load in generated questions", default=None)
    parser.add_argument("--saved_responses", help="Path to load in generated question-answer pairs", default=None)
    parser.add_argument("--benchmark", type=str, help="flag for running benchmark", default=None)
    parser.add_argument("--save_name", type=str, help="unique name to identify saved scores", default="no_name_specified")
    parser.add_argument('--get_benchmark_answers_only', help="flag for getting answers of benchmark questions ONLY without evaluation or question generation", action='store_true')
    parser.add_argument('--eval_1_only', help="flag for running evaluation with EVAL_1 model only", action='store_true')
    parser.add_argument('--eval_2_only', help="flag for running evaluation with EVAL_2 model only", action='store_true')
    parser.add_argument("--addr", type=str, help="Address for model service ip and port", default="127.0.0.1:8000")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_loader()

    if args.benchmark:
        persona_list = benchmark_personas
        saved_questions = args.benchmark
        saved_responses = args.saved_responses
    else:
        persona_list = eval(args.persona_list)
        saved_questions = args.saved_questions
        saved_responses = args.saved_responses

    if args.get_benchmark_answers_only:
        for i, persona in enumerate(persona_list):
            main_get_benchmark_answers(persona, args.model, args.model_name, saved_questions, saved_responses, args.addr)

    else:
        results = {}
        for i, persona in enumerate(persona_list):
            scores = main(persona, args.model, args.model_name, saved_questions, saved_responses, args.eval_1_only, args.eval_2_only, args.addr)
            results[persona] = scores["PersonaScore"]
            save_scores(args.save_name, scores, persona)
            logger.info(f'Done with {i + 1}/{len(persona_list)} personas')

        logger.info(results)
        logger.info("Evaluation Done!")
