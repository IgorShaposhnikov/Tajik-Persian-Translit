import os
import pandas as pd


def prepare_translation_datasets(data_path):
    with open(os.path.join(data_path, "train.trg"), "r", encoding="utf-8") as f:
        persian_text = f.readlines()
        persian_text = [text.strip("\n") for text in persian_text]

    with open(os.path.join(data_path, "train.src"), "r") as f:
        english_text = f.readlines()
        english_text = [text.strip("\n") for text in english_text]

    data = []
    for persian, tajik in zip(persian_text, english_text):
        data.append(["translate persian to tajik", persian, tajik])
        data.append(["translate tajik to persian", tajik, persian])

    train_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    # with open(os.path.join(data_path, "test.trg"), "r", encoding="utf-8") as f:
    #     persian_text = f.readlines()
    #     persian_text = [text.strip("\n") for text in persian_text]
    #
    # with open(os.path.join(data_path, "test.src"), "r") as f:
    #     english_text = f.readlines()
    #     english_text = [text.strip("\n") for text in english_text]

    data = []
    for persian, tajik in zip(persian_text, english_text):
        data.append(["translate persian to tajik", persian, tajik])
        data.append(["translate tajik to persian", tajik, persian])

    eval_df = pd.DataFrame(data, columns=["prefix", "input_text", "target_text"])

    return train_df, eval_df
