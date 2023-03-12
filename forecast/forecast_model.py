import torch 
import numpy as np
from torch import nn
from template import *
import torch.nn.functional as F

class forecast_with_text(nn.Module):
    def __init__(self, backbone, args):
        super().__init__()
        self.args = args
        self.backbone = backbone
        self.lin1 = nn.Linear(args.emb_size + args.num_lags, args.emb_size + args.num_lags)
        self.lin2 = nn.Linear(args.emb_size + args.num_lags, 1)
        self.act = nn.GELU()
    
    def forward(self, data): 
        # print(self.args.device)
        encoded = self.backbone(input_ids = data["input_ids"].to(self.args.device),
                                attention_mask = data["attn_mask"].to(self.args.device))
        CLS_encoded = encoded["pooler_output"]

        if self.args.num_lags > 0:
            CLS_encoded = torch.cat((CLS_encoded, data["rate_change_lags"].to(self.args.device)), dim = -1)
            ## NOTE: check the correctness of that 
        pred = self.lin1(CLS_encoded)


        ## TODO: add activation and more fancy stuff 
        pred = self.act(pred)
        pred = self.lin2(pred)
        return pred

class forecast_trainer(BaseEstimator):

    def step(self, data):


        if type(data["input_ids"]) is list: 
          data["input_ids"] = torch.tensor([t.tolist() for t in data["input_ids"]]).transpose(0, 1)
          data["attn_mask"] = torch.tensor([t.tolist() for t in data["attn_mask"]]).transpose(0, 1)
          if self.cfg.num_lags > 0:
            data["rate_change_lags"] = torch.tensor([t.tolist() for t in data["rate_change_lags"]]).transpose(0, 1)

      
        pred = self.model(data)
        ## TODO: change X here
        loss = self.criterion(pred.float(), data["rate_change"].to(self.cfg.device).float())
        if self.mode == "train":
            self.model.train()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()

        return {
            "loss": loss.detach().cpu().item(), 
            "pred": pred.detach().cpu(),
            "y": data["rate_change"]
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
        
        # overall_loss = self.criterion(preds, ys)
        MSEloss = mse_loss(ys, preds)
        MAEloss = mae_loss(ys, preds)
        print(f"==============The evaluation MSE loss is {MSEloss}==============")
        print(f"==============The evaluation MAE loss is {MAEloss}==============")
        return preds, ys, MSEloss, MAEloss


def mse_loss(pred, ref):
  loss = 0
  assert len(pred) == len(pred)
  for i in range(len(pred)):
    loss += (pred[i] - ref[i])**2
  return loss / len(pred)


def mae_loss(pred, ref):
  loss = 0
  assert len(pred) == len(pred)
  for i in range(len(pred)):
    loss += abs(pred[i] - ref[i])
  return loss / len(pred)
    





        


