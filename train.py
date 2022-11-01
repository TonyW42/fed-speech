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
from utils import load_model, save_model, make_if_not_exists, setup_seed

## train model
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def train(model, optimizer, train_dataloader, device, EPOCHS, ckpt_dir = None, load_model_path = None, start_epoch = 0):
  #model = model.to(device)
  if ckpt_dir is not None : 
    make_if_not_exists(ckpt_dir) ## added
  train_loss = []
  if load_model_path is not None:
    load_model(model, load_model_path, optimizer = optimizer)

  for epoch_i in range(start_epoch, EPOCHS):
    print(f'Beginning epoch {epoch_i + 1} of {EPOCHS}')

    total_train_loss = 0
    model.train()
    
    t0 = time.time()

    for step, batch in enumerate(train_dataloader):
      # print(f"step: {step}, number of objects: {len(gc.get_objects())}")
      # print(f"starting epoch {epoch_i+1} batch {step}/{int(train_size/BS)+1}")
      # b_input_ids = batch[0].to(device)
      # b_labels = batch[0].to(device)
      # b_masks = batch[1].to(device)

      model.zero_grad()
      outputs = model(    batch[0].to(device),
                          labels=batch[0].to(device), 
                          attention_mask = batch[1].to(device),
                          token_type_ids=None,
                          return_dict = True
                        )
      loss = outputs["loss"]  
      batch_loss = loss.detach().item() ## detach or not? 
      total_train_loss += batch_loss

      loss.backward()
      optimizer.step()

      model.zero_grad()

      torch.cuda.empty_cache()
    
    training_time = format_time(time.time() - t0)
    
    avg_train_loss = total_train_loss / len(train_dataloader)  
    if ckpt_dir is not None: 
      save_model(model, 
                 ckpt_dir, 
                 epoch = epoch_i, 
                 loss = avg_train_loss)

    gc.collect()
    torch.cuda.empty_cache()

    print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')



