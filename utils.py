import os 
import torch 
import transformers
import numpy as np
import random 
import math 
import statistics

def adjust_model_setting(model, UNFREEZE_LAYER):
  model.zero_grad()
  
  for parameter in model.parameters():
    parameter.requires_grad = False

  for i, m in enumerate(model.transformer.h):        
      #Only un-freeze the last n transformer blocks
      if i+1 > 12 - UNFREEZE_LAYER:
          for parameter in m.parameters():
              parameter.requires_grad = True 

  for parameter in model.transformer.ln_f.parameters():        
      parameter.requires_grad = True

  for parameter in model.lm_head.parameters():        
      parameter.requires_grad = True

## save model
def save_model(model, ckpt_path, epoch, loss, optimizer = None):
  if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
  ckpt_path_epoch = f"{ckpt_path}/model_epoch_{epoch+1}.pt"
  checkpoint = {
            'epoch': epoch, 
            'model': model.state_dict(), 
            'loss' : loss,
            'optimizer': optimizer.state_dict() if optimizer is not None else None, 
        }
  torch.save(checkpoint, ckpt_path_epoch)

## load model
def load_model(model, ckpt_path, optimizer = None):
  checkpoint = torch.load(ckpt_path)
  model.load_state_dict(checkpoint['model'])
  if optimizer is not None: 
      optimizer.load_state_dict(checkpoint['optimizer'])

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_if_not_exists(new_dir): 
    if not os.path.exists(new_dir): 
        os.system('mkdir -p {}'.format(new_dir))