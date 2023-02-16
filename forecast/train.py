import torch 
import numpy as np
import pandas as pd
import transformers 
from transformers import AutoModel, AutoTokenizer, get_scheduler
from forecast_model import *
from forecast_data import *
from torch.utils.data import Dataset,DataLoader, RandomSampler, \
                             SequentialSampler, random_split



def train(args):

    ## get model 
    backbone = AutoModel.from_pretrained(args.model_name)
    args.emb_size = backbone.config.hidden_size
    model = forecast_with_text(backbone, args).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ## get data
    trainloader, devloader, testloader = get_data(tokenizer, args)

    ## get optimizer
    ## NOTE: freeze backbone
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr) ## NOTE: weight decay
    num_training_steps = args.n_epochs * len(trainloader)
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    logger = None ## TODO: add logger to track progress
    criterion = torch.nn.MSELoss() if args.loss_fn == "MSE" else torch.nn.L1Loss()

    ## get classifier 
    trainer = forecast_trainer(
        model = model, 
        cfg = args,
        criterion = criterion, 
        optimizer = optimizer, 
        scheduler = scheduler, 
        device = args.device,
        logger = logger 
    )

    ## train 
    if args.mode == "train":
        trainer.train(args, trainloader, testloader)
