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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import random_split
# from run import TR_SIZE, tokenizer, MAXLEN, SPECIAL_TOKENS, BS, NUM_WORKER, DATA_NAME 
def load_data(TR_SIZE, tokenizer, MAXLEN, SPECIAL_TOKENS, BS, NUM_WORKER, DATA_NAME, prompt_type):
  ## load statements 
  data = dict()
  root_path = "data/statement"
  for root, dirs, files in os.walk(root_path, topdown=False):
      for name in files:
          if name != ".DS_Store":
              file_path = os.path.join(root, name)
          with open(file_path) as txt:
              txt_lines = txt.read()
              txt.close()
          txt_date = name.split(".")[0]
          data[txt_date] = txt_lines

  # print(data)

  statement_max_len = max([len(tokenizer.encode(data[key])) for key in data])
  short_data = dict()
  for key in data:
    if len(tokenizer.encode(data[key])) < (1024 - 10):
      short_data[key] = data[key]

  speeches = pd.read_csv("data/speech_with_description.csv")

  ## load speeches 
  def partition_speech(speech_data, keep_short = True, upper_limit = 150):
    speech = []
    oversized_count = 0
    for idx in range(0, speech_data.shape[0]):
      # result_tmp = []
      description = speech_data["description"].values[idx]
      consumption = speech_data["consumption"].values[idx]
      economic_activity = speech_data["economic_activity"].values[idx]
      inflation = speech_data["inflation"].values[idx]
      unemployment = speech_data["unemployment"].values[idx]

      text = speech_data["text"].values[idx]
      text = tokenize.sent_tokenize(text)
      count = 0
      # partitioned_text = []
      # print(len(text))
      while count < len(text):
        text_tmp = []
        word_count = 0
        while word_count < 100 and count < len(text):  ## every paragraph (partitioned) does not have over 200 words
          # print(count)
          word_count += len(text[count].split(" "))
          if keep_short and len(text_tmp) > 0:
            if word_count > upper_limit: break 
          text_tmp.append(text[count])
          # if (len(text[count].split(" ")) > 300): 
          #   # print(idx)
          #   oversized_count += 1
          count += 1
          # if (len(text[count].split(" ")) > 1000): print("problem!")
        text_tmp = ' '.join(text_tmp)
        speech.append({"description": description, "text": text_tmp,
                       "consumption" : consumption, "economic activity": economic_activity,
                       "inflation": inflation, "unemployment": unemployment})
      # result_tmp["text"] = partitioned_text
      # speech.append(result_tmp)
      if idx % 100 == 0:
        print(f"Done {idx}/{speech_data.shape[0]}")
    print(oversized_count)
    return speech 

  speech = partition_speech(speeches)

  ## data class for statement 
  class statement_data(Dataset):
    def __init__(self, data, tokenizer, randomize = True):
      self.text = data.values()
      self.tokenizer = tokenizer
      self.input_ids = []
      self.attn_masks = []
      self.description = []
      self.content = []
      for txt in self.text:
        self.description.append(txt[0])
        self.content.append(txt[1:len(txt)])
        input = SPECIAL_TOKENS["bos_token"] + txt[0] + \
                SPECIAL_TOKENS["sep_token"] + txt[1:len(txt)] + \
                SPECIAL_TOKENS["eos_token"] ## added
        input_dict = tokenizer(input, truncation=True, max_length = MAXLEN, 
                              padding = "max_length")
        self.input_ids.append(torch.tensor(input_dict['input_ids']))
        self.attn_masks.append(torch.tensor(input_dict['attention_mask']))


    def __len__(self):
      return(len(self.input_ids))
    
    def __getitem__(self, idx):
      description = self.description[idx]
      content = self.content[idx]
      return self.input_ids[idx], self.attn_masks[idx], description, content

  ## data class for speech 
  class speech_data(Dataset):
    def __init__(self, data, tokenizer, prompt_type, randomize = True):
      self.tokenizer = tokenizer
      self.data = data
      self.prompt_type = prompt_type

    def __len__(self):
      return(len(self.data))
    
    def __getitem__(self, idx):
      description = self.data[idx]["description"]
      text = self.data[idx]["text"]
      if self.prompt_type == "sentence":
        inputs = SPECIAL_TOKENS["bos_token"] + description + \
                  " Here is an excerpt from the speech: \n" + \
                  SPECIAL_TOKENS["sep_token"] + text + \
                  SPECIAL_TOKENS["eos_token"] 
      elif self.prompt_type == "word":
        inputs = ""
        for key in ["consumption", "economic activity", "inflation", "unemployment"]:
          inputs += f"{key}: {self.data[idx][key]} "
        inputs += SPECIAL_TOKENS["sep_token"] + text
      input_dict = self.tokenizer(inputs, truncation=True, max_length = MAXLEN, 
                                  padding = "max_length")
      input_ids = torch.tensor(input_dict['input_ids'])
      attn_masks = torch.tensor(input_dict['attention_mask'])

      return input_ids, attn_masks, description, text

  ## load dataset 
  dataset = None 
  if DATA_NAME == "speech":
      dataset = speech_data(speech, tokenizer, prompt_type)
  if DATA_NAME == "statement":
      dataset = statement_data(short_data, tokenizer)
  if dataset == "None":
      ValueError("Please choose either statement or speech")


  # Split into training and validation sets
  train_size = int(0.9 * len(dataset))
  val_size = len(dataset) - train_size

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  f'There are {train_size} samples for training, and {val_size} samples for validation testing'



  train_dataloader = DataLoader(
              train_dataset,  
              sampler = RandomSampler(train_dataset), # Sampling for training is random
              batch_size = BS,
              # num_workers = NUM_WORKER 
          )

  validation_dataloader = DataLoader(
              val_dataset, 
              sampler = SequentialSampler(val_dataset), # Sampling for validation is sequential as the order doesn't matter.
              batch_size = BS,
              # num_workers = NUM_WORKER 
          )
  return train_dataset, val_dataset, train_dataloader, validation_dataloader

