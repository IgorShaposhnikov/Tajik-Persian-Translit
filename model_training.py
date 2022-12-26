import logging
import pandas as pd
import torch.cuda
from simpletransformers.t5 import T5Model, T5Args
from simpletransformers.config import global_args
from prepare_data import prepare_translation_datasets


def train_model():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # train_df, eval_df = prepare_translation_datasets("/home/hel2x/programming/python/urfu-project/eryhjtklrjdtjklrdkljgdr/data/tg-fa")
    # train_df.to_csv("/home/hel2x/programming/python/urfu-project/eryhjtklrjdtjklrdkljgdr/data/train.tsv", sep="\t")
    # eval_df.to_csv("/home/hel2x/programming/python/urfu-project/eryhjtklrjdtjklrdkljgdr/data/eval.tsv", sep="\t")

    train_df = pd.read_csv(
        "/home/hel2x/programming/python/urfu-project/eryhjtklrjdtjklrdkljgdr/data/train.tsv", sep="\t").astype(str)
    eval_df = pd.read_csv(
        "/home/hel2x/programming/python/urfu-project/eryhjtklrjdtjklrdkljgdr/data/eval.tsv", sep="\t").astype(str)

    train_df["prefix"] = ""
    eval_df["prefix"] = ""

    model_args = T5Args()
    model_args.max_seq_length = 96
    model_args.train_batch_size = 20
    model_args.eval_batch_size = 20
    model_args.num_train_epochs = 1
    model_args.evaluate_during_training = False
    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_return_sequences = 1

    torch.cuda.memory_summary(device=None, abbreviated=False)

    model = T5Model("mt5", "google/mt5-base", args=model_args, use_cuda=True)
    model.train_model(train_df, eval_data=eval_df)
