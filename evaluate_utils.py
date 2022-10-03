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
from utils import make_if_not_exists, setup_seed

def prompt_response(val_dataset):
    ## get prompt-response dict
    prompt_response_dict = {}
    for j, samp in enumerate(val_dataset):
        _, _, prompt, text = samp
        if prompt not in prompt_response_dict: 
            prompt_response_dict[prompt] = [text, ]
        else:
            prompt_response_dict[prompt].append(text)
        if j% 500 == 0 or j == (len(val_dataset)-1): 
            print(f"Done {j}/{len(val_dataset)}")
    return prompt_response_dict

def macro_bleu_efficient(model, tokenizer, SPECIAL_TOKENS, device, val_dataset,  weights = [0.25, 0.25, 0.25, 0.25]):
  model.eval()
  prompt_response_dict = prompt_response(val_dataset)
#   print(prompt_response_dict) ## delete
  bleu_scores = []
  sep_id = tokenizer(SPECIAL_TOKENS["sep_token"])["input_ids"][0]
  for j, samp in enumerate(val_dataset):
    input_ids, attn_mask, prompt, ref = samp
    prompt_tokenized = tokenizer.encode(
        SPECIAL_TOKENS["bos_token"] + prompt + SPECIAL_TOKENS["sep_token"], 
        return_tensors = "pt").to(device)
    # print(prompt_tokenized)
    # break
    # prompt_ids = prompt_tokenized.unsqueeze(0)
    prompt_ids = prompt_tokenized
    references = prompt_response_dict[prompt]
    ref_text = []
    for ref in references:
      ref_text_raw = re.split(
        "\s|\W", # "(\s|\W)" to keep delimeter
        ref
        )
      ref_text_tmp = [tok for tok in ref_text_raw if tok != "" ]
      ref_text.append(ref_text_tmp)

    generated = model.generate(
        prompt_ids,
        do_sample = True, 
        max_length  = 300,
        min_length = 200,
        num_return_sequences = 1
    )
    # print(generated)
    generated_all = []
    for i, sample in enumerate(generated):
      sep_idx_2 = int((sample == sep_id).nonzero()[0])
      generated_all.append(sample[sep_idx_2:len(sample)]) ## sep_idx + 1? 
    generated_texts = []
    for i, gen_samp in enumerate(generated_all):
      generated_texts.append(tokenizer.decode(gen_samp, skip_special_tokens=True))
    generated_split_raw = [re.split("\s|\W", sent) for sent in generated_texts]
    generated_split = []
    for sent in generated_split_raw: 
      generated_split.append([l for l in sent if l != ""])
    # print(generated_split[0])
    bleu_score_samp = sentence_bleu(references = ref_text,
                                    hypothesis = generated_split[0],
                                    weights = weights,
                                    smoothing_function=SmoothingFunction().method1
                                    )
    bleu_scores.append(bleu_score_samp)
    # print(ref_text)
    # print(generated_split)
    # break 
    if j % 50 == 0: 
      print(f"Finished {j}/{len(val_dataset)}, current BLEU: {statistics.mean(bleu_scores)}")
  return bleu_scores

def semantic_similarity(model, tokenizer, SPECIAL_TOKENS, device, val_dataset):
  similarity_model = SentenceTransformer('stsb-roberta-large')
  prompt_response_dict = prompt_response(val_dataset)
  model.eval()
  similarities = []
  sep_id = tokenizer(SPECIAL_TOKENS["sep_token"])["input_ids"][0]
  for j, samp in enumerate(val_dataset):
    input_ids, attn_mask, prompt, ref = samp
    prompt_tokenized = tokenizer.encode(
        SPECIAL_TOKENS["bos_token"] + prompt + SPECIAL_TOKENS["sep_token"], 
        return_tensors = "pt").to(device)
    # print(prompt_tokenized)
    # break 
    # prompt_ids = prompt_tokenized.unsqueeze(0)
    prompt_ids = prompt_tokenized
    references = prompt_response_dict[prompt]

    generated = model.generate(
        prompt_ids,
        do_sample = True, 
        max_length  = 300,
        min_length = 200,
        num_return_sequences = 2
    )
    generated_all = []
    for i, sample in enumerate(generated):
      sep_idx_2 = int((sample == sep_id).nonzero()[0])
      generated_all.append(sample[sep_idx_2:len(sample)]) ## sep_idx + 1? 
    generated_texts = []
    for i, gen_samp in enumerate(generated_all):
      generated_texts.append(tokenizer.decode(gen_samp, skip_special_tokens=True))

    references_encoded = [similarity_model.encode(r, convert_to_tensor=True) for r in references]
    generated_encoded = [similarity_model.encode(g, convert_to_tensor=True) for g in generated_texts]


    list_of_similarities = []
    for r in references_encoded:
      for g in generated_encoded:
        cos_similarity = util.pytorch_cos_sim(r, g)
        list_of_similarities.append(float(cos_similarity))
    # print(list_of_similarities)
    similarities.append(max(list_of_similarities))  ## change to statistics.mean??

    # print(references)
    # print(generated_texts)
    # break 
    # break 
    if j % 50 == 0: 
      print(f"Finished {j}/{len(val_dataset)}, current similarity: {statistics.mean(similarities)}")
  return similarities

