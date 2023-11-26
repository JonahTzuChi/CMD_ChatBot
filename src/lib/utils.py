import re
from enum import Enum
import requests
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


def update_context(context: list, role: config.Roles, message: str) -> list:
    context.append({"role": role.value, "content": message})
    return context


def start():
    model = pick_model()
    print(f">Model = {model}")

    # Container to store chat history with one initial prompt to configure the conversation
    context = [
        {"role": Roles.SYSTEM, "content": "Be a friendly companion and offer casual chat."}
    ]
    tokens = 0  # accumulated tokens spent in this session

    try:
        while True:
            response_text, usage = chat(model, context)

            update_context(
                context=context, role=Roles.ASSISTANT, message=response_text
            )
            tokens += usage

            user_feedback = input(f"$_$: {response_text}\n^_^: ")

            if user_feedback == "q":
                break

            update_context(
                context=context, role=Roles.USER, message=user_feedback
            )
        print(f"Accumulated Token Usage: {tokens}")
    except OpenAIRequestError as req_err:
        print(f"Request Eror: {req_err}", flush=True)
    except OpenAIResponseParsingError as parsing_err:
        print(f"Parsing Error: {parsing_err}", flush=True)
    except KeyboardInterrupt:
        print(f"Accumulated Token Usage: {tokens}")
    except Exception as e:
        print("#$%^$#", str(e))
