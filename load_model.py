import logging
import sacrebleu
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args


def load_model():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    model_args = T5Args()
    model_args.max_length = 512
    model_args.length_penalty = 1
    model_args.num_beams = 10

    model = T5Model("mt5", "outputs", args=model_args, use_cuda=True)

    eval_df = pd.read_csv("data/eval.tsv", sep="\t").astype(str)

    sinhala_truth = [eval_df.loc[eval_df["prefix"] == "translate tajik to persian"]["target_text"].tolist()]
    to_sinhala = eval_df.loc[eval_df["prefix"] == "translate tajik to persian"]["input_text"].tolist()

    english_truth = [eval_df.loc[eval_df["prefix"] == "translate persian to tajik"]["target_text"].tolist()]
    to_english = eval_df.loc[eval_df["prefix"] == "translate persian to tajik"]["input_text"].tolist()

    # Predict
    sinhala_preds = model.predict(to_sinhala)

    eng_sin_bleu = sacrebleu.corpus_bleu(sinhala_preds, sinhala_truth)
    print("--------------------------")
    print("tajik to persian: ", eng_sin_bleu.score)

    english_preds = model.predict(to_english)

    sin_eng_bleu = sacrebleu.corpus_bleu(english_preds, english_truth)
    print("persian to tajik: ", sin_eng_bleu.score)
