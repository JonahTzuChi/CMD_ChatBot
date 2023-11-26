import re
import requests
import config
from lib.custom_exception import OpenAIRequestError, OpenAIResponseParsingError

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
        break # valid input
    return config.MODELS[idx-1]["name"]

def start():
    model = pick_model()
    print(f">Model = {model}")