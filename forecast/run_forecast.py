
import os 
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch 
import argparse
from train import *


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--data_path', type=str, default="data/speech_with_description.csv")
parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--output_dir', type=str, default="./results")
parser.add_argument('--bs', type=int, default=16)
parser.add_argument('--model_name', type=str, default="bert-base-uncased")
parser.add_argument('--n_epochs', type=int, default=1) ## change to 4
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--tr_size', type=float, default=0.7)
parser.add_argument('--dev_size', type=float, default=0.1)
parser.add_argument('--max_len', type=int, default=350)
parser.add_argument('--num_unfreeze', type=int, default=6)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--data', type=str, default="speech")
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--loss_fn', type=str, default="MSE")
parser.add_argument('--num_lags', type=int, default=0)

args = parser.parse_args()


# os.environ['TOKENIZERS_PARALLELISM']= 'True'



if __name__ == "__main__":
    train(args)







