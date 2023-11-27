import os
import re
from enum import Enum
import requests
import pandas as pd
import PyPDF2
import numpy as np
import json

import config
from lib.custom_exception import OpenAIRequestError, OpenAIResponseParsingError


class Roles(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


def pick_model() -> str:
    print("Pick the model by enter the corresponding number")
    for idx, model in enumerate(config.MODELS, 1):
        print(
            f'[{idx: 2}]\t{model["name"]:16s} InputToken: {model["in"]:8s}, OutputToken: {model["out"]:8s}'
        )
    idx = 1
    while True:
        choice = input()
        # detect none numeric
        if re.search(r"[^0-9+-]", choice):
            print("Numbers only")
            continue
        idx = int(choice)
        # detect out of bound
        if idx > len(config.MODELS) or idx < 1:
            print(f"Expect 1 to {len(config.MODELS)}")
            continue
        break  # valid input
    return config.MODELS[idx - 1]["name"]


def chat(model: str, messages: list) -> tuple[str, int]:
    try:
        URL = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        data = {"model": model, "messages": messages}
        response = requests.post(URL, headers=headers, json=data)

        response.raise_for_status()  # Raises HTTPError for 4xx/5xx codes

        response_json = response.json()

        if "error" in response_json.keys():
            err = response_json.get("error")
            raise OpenAIRequestError(f"{err['type']}, message:{err['message']}")

        response_text = response_json.get("choices")[0].get("message")["content"]
        usage = response_json.get("usage")["total_tokens"]

        return response_text, usage
    except requests.exceptions.HTTPError as http_err:
        raise OpenAIRequestError(f"HTTP error: {http_err}")
    except KeyError as key_err:
        raise OpenAIResponseParsingError(f"Response parsing error: {key_err}")
    except Exception as e:
        raise OpenAIRequestError(f"Other error: {e}")


def update_context(
    context: list[dict[str, str]], role: config.Roles, message: str
) -> list:
    context.append({"role": role.value, "content": message})
    return context


def dynamic_context_management(
    model: str, context: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    Q0 = context[0]

    COMMAND = {
        "role": Roles.SYSTEM.value,
        "content": "Please summarize our past conversation. I expect to use the summarized version for subsequent conversation, therefore the result should contain sufficient information for you to hold the subsequent conversation in a context aware manner. I also expect the summary to be 50 to 75% shorter than the sum of our past conversation.",
    }
    context.append(COMMAND)
    response_text, usage = chat(model, context)
    newContext = [
        Q0,
        {"role": Roles.ASSISTANT.value, "content": response_text},
    ]
    return newContext, usage


def sanitize_filename(filename: str) -> str:
    INVALID_CHARACTERS = '<>:"/\\|?*'
    for char in INVALID_CHARACTERS:
        filename = filename.replace(char, "_")
    # Truncate the filename if it's too long
    MAX_LENGTH = 64
    return filename[:MAX_LENGTH]


def export_chat_history(
    filename: str, chat_history: list[dict[str, str]], create_new: bool
) -> str:
    """
    COLUMNS: Role, Content
    FORMAT : Start with ^, end with $, and separated by _______
    """
    filename = sanitize_filename(filename)
    flag = "w" if create_new else "a"
    EXPORT_PATH = f"./output/{filename}.txt"
    with open(EXPORT_PATH, flag) as f:
        for chat in chat_history[:]:
            f.write(f"^{chat['role']}_______{chat['content']}$\n")
    return EXPORT_PATH


def read_starndard_text_file(filename: str) -> str:
    data = f"FileName:{filename}"
    with open(filename, "r") as reader:
        lines = reader.readlines()
        data += "".join(lines)

    return data


def read_spreadsheet(filename: str) -> str:
    data = f"FileName:{filename}"
    workbook = pd.read_excel(filename, sheet_name=None)
    for sheetname in workbook.keys():
        df = workbook[sheetname]
        data += f"\nSheetName:{sheetname}" + df.to_json()
    return data


def read_pdf(filename: str) -> str:
    data = f"FileName:{filename}"
    with open(filename, "rb") as fileObj:
        pdfReader = PyPDF2.PdfReader(fileObj)
        pages = pdfReader.pages
        print(len(pages))
        for page in pages:
            txt = page.extract_text()
            data += f"\nnewPage:" + txt
    return data


def read_npy_file(filename: str) -> str:
    data = f"FileName:{filename}"
    data = np.load(filename, encoding="ASCII")
    data_list = data.tolist()
    json_data = json.dumps(data_list)
    return data + f"\n{json_data}"


def pickTextFileReader(extension: str):
    if extension in [
        "txt",
        "csv",
        "py",
        "js",
        "json",
        "sql",
        "yml",
        "env",
        "h",
        "cpp",
        "cc",
        "java",
        "cs",
        "html",
        "php",
        "rs",
        "go",
        "Dockerfile",
    ]:
        return read_starndard_text_file
    if extension in ["xls", "xlsx"]:
        return read_spreadsheet
    if extension in ["pdf"]:
        return read_pdf
    if extension in ["npy"]:
        return read_npy_file
    raise "Invalid File Extension"


def readFile(filename: str) -> str:
    try:
        if ~os.path.exists(filename):
            return "FileIOError"

        extension = filename.split(".")[-1]
        fileReader = pickTextFileReader(extension)

        return fileReader(filename)
    except Exception as e:
        print(str(e), flush=True)
        raise e


def start():
    model = pick_model()
    print(f">Model = {model}")

    # Container to store chat history with one initial prompt to configure the conversation
    context = [
        {
            "role": Roles.SYSTEM.value,
            "content": "Be a friendly companion and offer casual chat.",
        }
    ]
    global_context = []

    Q0 = input("How can I help you?\n")  # Will be used as filename
    update_context(context=context, role=Roles.USER, message=Q0)

    tokens = 0  # running tokens
    accumulated_tokens = 0  # accumulated tokens spent in this session
    try:
        while True:
            response_text, usage = chat(model, context)

            update_context(context=context, role=Roles.ASSISTANT, message=response_text)
            tokens += usage

            if tokens > config.THROTTLE_THRESHOLD:
                accumulated_tokens += tokens
                global_context.extend(context[1:])
                context, tokens = dynamic_context_management(model, context)

            if accumulated_tokens > config.TERMINATION_THRESHOLD:
                print(
                    "$$$It is likely you have reach or exceeded the termination threshold!"
                )
                break

            user_feedback = input(f"$_$: {response_text}\n^_^: ")
            user_feedback = user_feedback.strip()

            if re.search("^--file ", user_feedback) is not None:
                filename = user_feedback.replace("--file ", "")
                user_feedback = readFile(filename)
                while user_feedback == "FIleIOError":
                    print(f"File cannot be found at {filename}")
                    user_feedback = input()
                    user_feedback = user_feedback.strip()
                    if user_feedback == "q":
                        break

            if user_feedback == "q":
                break

            update_context(context=context, role=Roles.USER, message=user_feedback)

        accumulated_tokens += tokens
        print(f"Accumulated Token Usage: {accumulated_tokens}")
        global_context.extend(context)
        export_filename = export_chat_history(Q0, global_context, create_new=True)
        print(f"Exported to {export_filename}")
    except OpenAIRequestError as req_err:
        print(f"Request Eror: {req_err}", flush=True)
    except OpenAIResponseParsingError as parsing_err:
        print(f"Parsing Error: {parsing_err}", flush=True)
    except KeyboardInterrupt:
        accumulated_tokens += tokens
        print(f"Accumulated Token Usage: {accumulated_tokens}")
        global_context.extend(context)
        export_filename = export_chat_history(Q0, global_context, create_new=True)
        print(f"Exported to {export_filename}")
    except Exception as e:
        print("#$%^$#", str(e))
