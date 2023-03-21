import pandas as pd
import numpy as np
import torch 
from datetime import datetime, timedelta
from fredapi import Fred
from torch.utils.data import Dataset,DataLoader, RandomSampler, \
                             SequentialSampler, random_split

class forecast_data(Dataset):
    def __init__(self, speech_df, tokenizer, args, randomize = True):
        self.args = args
        # fred = Fred(api_key='99c16d0eb16121bf66ccf5a4965f974c')
        sp500 = pd.read_csv("data/sp500.csv")
        sp500.columns = [s.strip() for s in sp500.columns] ## strip white spaces
        close = sp500["Close"]
        Open = sp500["Open"]
        close_d = [100*(np.log(close[i]) - np.log(Open[i])) for i in range(len(close))]
        # close_d.append(np.nan)
        sp500["close_d"] = close_d
        
        # self.sp500["close_d_pct"] = [np.nan].extend([(close[i+1] - close[i])/close[i] for i in range(len(close)-1)])
        # rate = np.empty(len(speech_df))
        rate_change = np.empty(len(speech_df))
        # speech_df["rate_change_tmr"] = np.nan
        rate_change_l1 = np.empty(len(speech_df))
        rate_change_l2 = np.empty(len(speech_df))
        rate_change_l3 = np.empty(len(speech_df))
        rate_change_l4 = np.empty(len(speech_df))
        for i in range(0, len(speech_df)):
            date = speech_df["date"][i]
            date = datetime.strptime(str(date), "%Y%m%d")
            date_tmp = date
            while date_tmp.strftime("%m/%d/%y") not in sp500["Date"].values:
                date_tmp += timedelta(days=1)
            row_number = [i for i in range(len(close)) if sp500["Date"][i] == date_tmp.strftime("%m/%d/%y")][0]
            rate_tmp = sp500["close_d"].values[row_number]
            # print(rate_tmp)
            rate_change[i] = rate_tmp

            rate_change_l1[i] = sp500["close_d"].values[row_number+1]
            rate_change_l2[i] = sp500["close_d"].values[row_number+2]
            rate_change_l3[i] = sp500["close_d"].values[row_number+3]
            rate_change_l4[i] = sp500["close_d"].values[row_number+4]
            # print(i)
            # if i == 0:
            #   print(speech_df["date"][i])
            #   print(rate_change[i])
            #   print(rate_change_l1[i])
            #   print(rate_change_l2[i])
            #   print(rate_change_l3[i])
            #   print(rate_change_l4[i])
            if i % 100 == 0 : print(f"done {i}/{len(speech_df)}")

        # speech_df["rate"] = np.nan
        # speech_df["rate_change"] = rate_change
        # speech_df["rate_change_l1"] = rate_change_l1
        # speech_df["rate_change_l2"] = rate_change_l2
        # speech_df["rate_change_l3"] = rate_change_l3
        # speech_df["rate_change_l4"] = rate_change_l4
        
        # self.speech_df = speech_df
        self.text = speech_df["text"].tolist()
        self.rate_change = rate_change
        self.rate_change_l1 = rate_change_l1
        self.rate_change_l2 = rate_change_l2
        self.rate_change_l3 = rate_change_l3
        self.rate_change_l4 = rate_change_l4

        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i):
        tokenized = self.tokenizer(self.text[i], truncation=True, padding ="max_length")

        tokenized["rate_change"] = self.rate_change[i]
        tokenized["rate_change_lags"] = [self.rate_change_l1[i], self.rate_change_l2[i], self.rate_change_l3[i], self.rate_change_l4[i]]
        tokenized["rate_change_lags"] = tokenized["rate_change_lags"][:self.args.num_lags]
        return {
          "input_ids" : tokenized["input_ids"],
          "attn_mask" : tokenized["attention_mask"],
          "rate_change": tokenized["rate_change"],
          "rate_change_lags" : tokenized["rate_change_lags"]
        }
        # a = dict()
        # a["input_ids"] = [0, 1, 0]
        # return a

def get_data(tokenizer, args):
    speech_df = pd.read_csv(args.data_path)
    # if args.load_tokenized_data == "false":
    #   dataset = forecast_data(speech_df, tokenizer, args)
    #   torch.save(dataset, "forecast/dataset.pt")
    # else: 
    #   dataset = torch.load("forecast/dataset.pt")
    # print(len(dataset[:100]))
    # assert args.tr_size + args.dev_size < 1

    train_size = int(args.tr_size * len(speech_df))
    dev_size = int(args.dev_size * len(speech_df))
    test_size = len(speech_df) - train_size - dev_size
    if args.load_tokenized_data == "false":
      test_dataset = forecast_data(speech_df.iloc[:test_size, ].reset_index(drop=True), tokenizer, args)
      dev_dataset = forecast_data(speech_df.iloc[test_size:(test_size + dev_size), ].reset_index(drop=True), tokenizer, args)
      train_dataset = forecast_data(speech_df.iloc[(test_size + dev_size):, ].reset_index(drop=True), tokenizer, args)
      torch.save({"test": test_dataset, "train": train_dataset, "dev": dev_dataset}, "forecast/dataset.pt")
    else: 
      datasets = torch.load("forecast/dataset.pt")
      test_dataset = datasets["test"]
      dev_dataset = datasets["dev"]
      train_dataset = datasets["train"]

      train_dataset.args = args
      dev_dataset.args = args
      test_dataset.args = args


    train_dataloader = DataLoader(
              train_dataset,  
              sampler = RandomSampler(train_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )

    dev_dataloader = DataLoader(
              dev_dataset,  
              sampler = RandomSampler(dev_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )
    
    test_dataloader = DataLoader(
              test_dataset,  
              sampler = RandomSampler(test_dataset), # Sampling for training is random
              batch_size = args.bs,
              # num_workers = NUM_WORKER 
          )
    return train_dataloader, dev_dataloader, test_dataloader

