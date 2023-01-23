import torch 
import numpy as np
from torch import nn
from forecast.template import *

class forecast_with_text(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.lin = nn.Linear(args.emb_size, 1)
    
    def forward(self, data):
        encoded = self.backbone(input_ids = data["input_ids"].to(self.args.device),
                                attention_mask = data["attn_mask"].to(self.args.device))
        CLS_encoded = encoded["pooler_output"]
        pred = self.lin(CLS_encoded)
        ## TODO: add activation and more fancy stuff 
        return pred

class forecast_trainer(BaseEstimator):

    def step(self, data):
        pred = self.model(data)
        ## TODO: change X here
        loss = self.criterion(pred, data["y"].to(self.cfg.device))
        if self.mode is "train":
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.detach().cpu().item(), 
            "pred": pred.detach().cpu(),
            "y": data["y"]
        }

    def _eval(self, evalloader):
        self.model.eval()
        tbar = tqdm(evalloader, dynamic_ncols=True)
        eval_loss = []
        ys = []
        preds = []

        for data in tbar: 
            ret_step = self.step(data)   ## y: [bs, seq_len]
            ys.extend(ret_step["y"])
            preds.extend(ret_step["pred"])
        
        overall_loss = self.criterion(preds, ys)
        print("The evaluation loss is {overall_loss}")
        return preds, ys








        














