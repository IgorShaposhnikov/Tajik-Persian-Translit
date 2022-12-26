import os
import torch
import model_training
from load_model import load_model

d = []


def main():
    ans = input('Choose:\n1) Обучить модель на основе данных /data/fa-tg/\n2) Оценить модель\n0) Выход')
    if ans == '1':
        model_training.train_model()
    elif ans == '2':
        load_model()
    elif ans == '3':
        raise SystemExit()
    else:
        main()


if __name__ == '__main__':
    main()
