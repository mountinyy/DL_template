import argparse
import torch
import random
import numpy as np
from omegaconf import OmegaConf
from train.train import train

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", required=True, default="bert-base-cased", choices=["bert-base-cased", "roberta-base", "cardiffnlp/twitter-roberta-base-sentiment-latest"], help="pretrained model name form huggingface")
    arg_parser.add_argument("--wandb_name", default=None, help="wandb name for run")

    return arg_parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = get_args()
    conf = OmegaConf.load('./config.yaml')
    set_seed(conf.common.seed)
    train(conf, args)



