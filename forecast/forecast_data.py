import pandas as pd
import numpy as np
import torch 
from torch.utils.data import Dataset,DataLoader, RandomSampler, \
                             SequentialSampler, random_split

class forecast_data(Dataset):
    def __init__(self, speech_df, tokenizer, randomize = True):
        speech_df["rate"] = np.nan
        speech_df["rate_change"] = np.nan
        speech_df["rate_change_tmr"] = np.nan
        speech_df["rate_change_l1"] = np.nan
        speech_df["rate_change_l2"] = np.nan
        speech_df["rate_change_l3"] = np.nan
        speech_df["rate_change_l4"] = np.nan
        for i in range(0, len(speech_df)):
            ## TODO: give sp500 value 
            date = self.speech_df["date"][i]
            speech_df["rate"][i] = np.nan
            speech_df["rate_change"][i] = np.nan
            speech_df["rate_change_tmr"][i] = np.nan
            speech_df["rate_change_l1"][i] = np.nan
            speech_df["rate_change_l2"][i] = np.nan
            speech_df["rate_change_l3"][i] = np.nan
            speech_df["rate_change_l4"][i] = np.nan

        self.speech_df = speech_df
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.speech_df)
    
    def __getitem__(self, i):
        tokenized = self.tokenizer(self.speech_df["text"][i])
        tokenized["rate"] = self.rate[i]
        tokenized["rate_change"] = self.rate_change[i]
        tokenized["rate_change_tmr"] = self.rate_change_tmr[i]
        # tokenized["rate_change_l1"] = self.rate_change_l1[i]
        # tokenized["rate_change_l2"] = self.rate_change_l2[i]
        # tokenized["rate_change_l3"] = self.rate_change_l3[i]
        # tokenized["rate_change_l4"] = self.rate_change_l4[i]
        tokenized["rate_change_lags"] = [self.speech_df[f"rate_change_l{i}"] for i in range(1, self.args.num_lags+1)]
        return tokenized 
        

def get_data(tokenizer, args):
    speech_df = pd.read_csv(args.data_path)
    dataset = forecast_data(speech_df, tokenizer, args)
    assert args.tr_size + args.dev_size < 1

    train_size = int(args.tr_size * len(dataset))
    dev_size = int(args.dev_size * len(dataset))
    test_size = len(dataset) - train_size - dev_size
    train_dataset, dev_dataset, test_dataset = random_split(dataset, [train_size, dev_size, test_size])

    train_dataloader = DataLoader(
              train_dataset,  
              sampler = RandomSampler(train_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )

    dev_dataloader = DataLoader(
              dev_dataset,  
              sampler = RandomSampler(train_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )
    
    test_dataloader = DataLoader(
              test_dataset,  
              sampler = RandomSampler(train_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )
    return train_dataloader, dev_dataloader, test_dataloader



