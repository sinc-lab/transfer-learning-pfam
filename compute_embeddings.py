"""Precompute ESM embeddings to speed up training"""
from tqdm import tqdm 
import os
import pickle 
import torch as tr
import argparse

from utils import read_original_data

parser = argparse.ArgumentParser()
parser.add_argument("-o", default="data/")
parser.add_argument("-i")
parser.add_argument("--device", default="cuda")

args = parser.parse_args()

max_len = 1022
if not os.path.isdir(args.o):
    os.mkdir(args.o)

# load ESM 
emb_model, alphabet = tr.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
emb_model.eval()
emb_model.to(args.device)
batch_converter = alphabet.get_batch_converter()

data = read_original_data(args.i)
    
for item in tqdm(range(len(data))):
    seq_name = data.iloc[item].sequence_name
    cache_file = f"{args.o}{seq_name.replace('/', '-')}.pk"
    if os.path.isfile(cache_file):
        continue
    # Crop larger domains to a center window
    seq = data.iloc[item].sequence
    label = data.iloc[item].family_id

    center = len(seq)//2
    start = max(0, center - max_len//2)
    end = min(len(seq), center + max_len//2)
    seq = seq[start:end] 
        
    x = [(0, seq)]
    try:
        with tr.no_grad():
            _, _, tokens = batch_converter(x)
            emb = emb_model(tokens.to(args.device), repr_layers=[33], 
            return_contacts=True)["representations"][33].detach().cpu() 
    except:
        print(seq_name, len(seq))
        raise
    emb = emb.permute(0,2,1)
    pickle.dump([emb.half(), label], open(cache_file, "wb"))
