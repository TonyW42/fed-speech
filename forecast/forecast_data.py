import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader, RandomSampler, \
                             SequentialSampler, random_split

class forecast_data(Dataset):
    def __init__(self, speech_df, tokenizer, randomize = True):
        self.speech_df = speech_df
        self.tokenizer = tokenizer
        self.rate = []
        self.rate_change = []
        self.rate_change_tmr = []
        self.rate_change_l1 = []
        self.rate_change_l2 = []
        self.rate_change_l3 = []
        self.rate_change_l4 = []
        for i in range(0, len(speech_df)):
            ## TODO: give sp500 value 
            date = self.speech_df["date"][i]
            self.rate[i] = None
            self.rate_change[i] = None 
            self.rate_change_tmr[i] = None
            self.rate_change_l1[i] = None 
            self.rate_change_l2[i] = None 
            self.rate_change_l3[i] = None 
            self.rate_change_l4[i] = None 
    
    def __len__(self):
        return len(self.speech_df)
    
    def __getitem__(self, i):
        tokenized = self.tokenizer(self.speech_df["text"][i])
        tokenized["rate"] = self.rate[i]
        tokenized["rate_change"] = self.rate_change[i]
        tokenized["rate_change_tmr"] = self.rate_change_tmr[i]
        tokenized["rate_change_l1"] = self.rate_change_l1[i]
        tokenized["rate_change_l2"] = self.rate_change_l2[i]
        tokenized["rate_change_l3"] = self.rate_change_l3[i]
        tokenized["rate_change_l4"] = self.rate_change_l4[i]
        return tokenized 
        

def get_data(args):
    
    pass 



