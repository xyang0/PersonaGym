import time
import requests
import openai
from openai import OpenAI
import numpy as np
from together import Together
import anthropic

from api_keys import *


def run_model(
                    input_prompt = None,
                    persona = None,
                    model_card = 'gpt-3.5-turbo',
                    temperature = 0.9, 
                    top_p = 0.9,
                    max_tokens = 3000,
                    message = None,
                    system = None,
                    addr = "127.0.0.1:8000"
                ):
    if "gpt" in model_card:
        return openai_chat_gen(input_prompt, persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens=max_tokens, message=message, system=system)
    elif "claude" in model_card:
        return claude_chat_gen(input_prompt, persona=persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    elif "open_character" in model_card:
        return open_character_llama3_instruct_chat_gen(input_prompt, persona=persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens=max_tokens, addr=addr)
    elif "llama" in model_card:
        return llama_chat_gen(input_prompt, persona=persona, model_card=model_card, temperature=temperature, top_p=top_p, max_tokens=max_tokens)


def openai_chat_gen(input_prompt = None,
                    persona = None,
                    apikey = OPENAI_API_KEY,
                    model_card = 'gpt-3.5-turbo',
                    temperature = 0.9, 
                    top_p = 0.9,
                    max_tokens = 4000,
                    max_attempt = 3,
                    time_interval = 2,
                    system = None,
                    message = None,
                   ):

    client = OpenAI(api_key=apikey)

    if not message:
        if persona:
            persona_prompt = f"Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
            message=[{"role": "system", "content": persona_prompt},
                    {"role": "user", "content": input_prompt}]

        else:
            if system:
                message=[{"role": "system", "content": system},
                         {"role": "user", "content": input_prompt}]
            else:
                message=[{"role": "user", "content": input_prompt}]

    while max_attempt > 0:
        try:
            response = client.chat.completions.create(
                model= model_card,
                messages = message,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
            )
            return response.choices[0].message.content

        except Exception as e:

            print('Exception Raised: ', e)

            max_attempt -= 1
            time.sleep(time_interval)

            print('Retrying left: ', max_attempt)

    return 'Error'


def claude_chat_gen(input_prompt,
                    persona = None,
                    apikey = CLAUDE_API_KEY,
                    model_card = 'claude-3-haiku-20240307',
                    temperature = 0, 
                    max_tokens = 4000,
                    max_attempt = 3,
                    time_interval = 5
                   ):

    assert (type(input_prompt) == str
            ), "claude api does not support batch inference."

    client = anthropic.Anthropic(api_key=apikey)

    if persona:
        persona_prompt = f"Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."

    message=[{"role": "user", "content": input_prompt}]

    while max_attempt > 0:

        try:
            if persona:
                response = client.messages.create(
                    model= model_card,
                    system = persona_prompt,
                    messages = message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text
            else:
                response = client.messages.create(
                    model= model_card,
                    messages = message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text

        except Exception as e:

            print('Exception Raised: ', e)

            max_attempt -= 1
            time.sleep(time_interval)

            print('Retrying left: ', max_attempt)

    return 'As an AI Model I cannot answer'


def llama_chat_gen(input_prompt,
                   persona = None,
                   apikey = LLAMA_API_KEY,
                   model_card = 'meta-llama/Meta-Llama-3-70B',
                   temperature = 0.9, 
                   top_p = 0.9,
                   max_attempt = 3,
                   time_interval = 5
                  ):

    assert (type(input_prompt) == str
            ), "openai api does not support batch inference."

    client = Together(api_key=apikey)

    if persona:
        persona_prompt = f"Adopt the identity of {persona}. Answer the questions while staying in strict accordance with the nature of this identity."
        message=[{"role": "system", "content": persona_prompt},
                 {"role": "user", "content": input_prompt}]
    else:
        message=[{"role": "user", "content": input_prompt}]

    while max_attempt > 0:

        try:
            response = client.chat.completions.create(
                model= model_card,
                messages = message,
                temperature=temperature,
                top_p = top_p,
            )
            return response.choices[0].message.content

        except Exception as e:

            print('Exception Raised: ', e)

            max_attempt -= 1
            time.sleep(time_interval)

            print('Retrying left: ', max_attempt)

    return 'Error'


def open_character_llama3_instruct_chat_gen(input_prompt,
                                            persona = None,
                                            model_card = "open_character_llama3_instruct",
                                            temperature = 0.9, 
                                            top_p = 0.9,
                                            max_tokens = 8192,
                                            max_new_tokens = 1024,
                                            addr = "127.0.0.1:8000"):

    assert (type(input_prompt) == str
            ), "does not support batch inference."

    infer_model = BaseModelConnection(addr)

    if persona:
        sys_prompt = "You are an AI character with the following Persona.\n\n"
        sys_prompt += f"# Persona\n{persona}\n\n"
        sys_prompt += "Please stay in your character and keep in compliance with the above Persona and Character Profile. "
        sys_prompt += "Be helpful and harmless to the user\'s requests."
    else:
        sys_prompt = "You are a helpful assistant. Be helpful and harmless to the user\'s requests."

    message = [{"role": "system", "content": sys_prompt},
               {"role": "user", "content": input_prompt}]

    data = {"messages": message, "model": "meta-llama/Meta-Llama-3-8B-Instruct", "max_tokens": max_new_tokens, "temperature": temperature, "stop": ["<|eot_id|>"]}
    response = requests.post("http://" + addr + "/v1/chat/completions", json=data)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        return "Error"
