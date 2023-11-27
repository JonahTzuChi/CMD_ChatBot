import os
import numpy as np
import pandas as pd
import PyPDF2
import json


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
        "md",
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
        if not os.path.exists(filename):
            return "FileIOError"

        extension = filename.split(".")[-1]
        fileReader = pickTextFileReader(extension)

        data = fileReader(filename)
        return data
    except Exception as e:
        print(str(e), flush=True)
        raise e
