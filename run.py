import pandas as pd
import numpy as np
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
from torch.utils.data import Dataset,DataLoader, RandomSampler, \
                             SequentialSampler, random_split
import random
import time
import datetime
import gc
import re
from nltk import tokenize
import nltk
import nltk.translate.bleu_score
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
nltk.download('punkt')
import statistics
from sentence_transformers import SentenceTransformer, util
import argparse
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForCausalLM
from data import load_data
from train import train
from utils import make_if_not_exists, setup_seed, load_model
from evaluate_utils import semantic_similarity, macro_bleu_efficient
import statistics

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='wnut17')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default="./results")
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--model_name', type=str, default="google/canine-s")
parser.add_argument('--n_epochs', type=int, default=1) ## change to 4
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--tr_size', type=float, default=0.8)
parser.add_argument('--max_len', type=int, default=350)
parser.add_argument('--num_unfreeze', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--data', type=str, default="speech")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--start_epoch', type=int, default=0)

args = parser.parse_args()

LR = args.lr
TR_SIZE = args.tr_size
MAXLEN = args.max_len                ## small for speeches
UNFREEZE_LAYER = args.num_unfreeze
EPOCHS = args.n_epochs
EPS = args.eps
SEED = args.seed
BS = args.bs #
SPECIAL_TOKENS = {
    'bos_token' : '<|BOS|>', 
    'eos_token' : '<|EOS|>', 
    'pad_token' : '<|PAD|>',
    'sep_token' : '<|SEP|>',
    'wrp_token' : '<|WP|>', 
    'unk_token' : '<|UNK|>',
    'rsp_token' : '<|RESPONSE|>'
}
NUM_WORKER = args.num_workers
DATA_NAME = args.data

os.environ['TOKENIZERS_PARALLELISM']= 'True'

## load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2', 
                                          bos_token=SPECIAL_TOKENS['bos_token'], 
                                          eos_token=SPECIAL_TOKENS['eos_token'], 
                                          pad_token=SPECIAL_TOKENS['pad_token'],
                                          unk_token=SPECIAL_TOKENS['unk_token'],
                                          sep_token=SPECIAL_TOKENS['sep_token'])

## load model
configuration = AutoConfig.from_pretrained('gpt2', output_hidden_states=False, 
                                           bos_token_id=tokenizer.bos_token_id,
                                           eos_token_id=tokenizer.eos_token_id,
                                           sep_token_id=tokenizer.sep_token_id,
                                           unk_token_id=tokenizer.unk_token_id,
                                           pad_token_id=tokenizer.pad_token_id)

# ## load data 
train_dataset, val_dataset, train_dataloader, test_dataloader = load_data(TR_SIZE, tokenizer, MAXLEN, SPECIAL_TOKENS, BS, NUM_WORKER, DATA_NAME)

## load model 
model = AutoModelForCausalLM.from_pretrained("gpt2", config=configuration)
model.resize_token_embeddings(len(tokenizer))
model.to(device)


if args.load_path is not None:
    model = load_model(model, ckpt_path = load_model)
    model.to(device)


## freeze layers 
for parameter in model.parameters():
    parameter.requires_grad = False

for i, m in enumerate(model.base_model.h):        
    #Only un-freeze the last n transformer blocks
    if i+1 > 12 - UNFREEZE_LAYER:
        for parameter in m.parameters():
            parameter.requires_grad = True 

for parameter in model.base_model.ln_f.parameters():        
    parameter.requires_grad = True

for parameter in model.lm_head.parameters():        
    parameter.requires_grad = True

## load optimizer 
optimizer = torch.optim.AdamW(model.parameters(), lr = LR, eps = EPS)

## load saved model
if args.load_path is not None:
    load_model(model, args.load_path) ## check this 

## train if mode == "train"
if args.mode == "train":
    train(model = model, optimizer = optimizer, train_dataloader = train_dataloader,
          device = device, EPOCHS = EPOCHS, ckpt_dir = args.save_dir,
          load_model_path = args.load_path, start_epoch = args.start_epoch)

## do evaluation 
bleu_scores = macro_bleu_efficient(model, tokenizer, SPECIAL_TOKENS, device, val_dataset)
similarity_scores = semantic_similarity(model, tokenizer, SPECIAL_TOKENS, device, val_dataset)

## do something with these scores

if __name__ == "__main__":
    print(f"BLEU Score : {statistics.mean(bleu_scores)}")
    print(f"Similarity Score : {statistics.mean(similarity_scores)}")
    torch.save(torch.tensor(bleu_scores), "bleu.pt")
    torch.save(torch.tensor(similarity_scores), "sim_score.pt")








