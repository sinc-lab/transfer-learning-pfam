import os 
import torch as tr 
from sklearn.metrics import accuracy_score

from dataset import ProtDataset, pad_batch
from torch.utils.data import DataLoader
from tlprotcnn import TLProtCNN

TEST_PATH = "data/Clustered_data/test/"
CACHE_PATH = "data/"
BATCH_SIZE = 128
DEVICE = "cuda"
categories = [item.strip() for item in open("data/categories.txt")]

# trained model weights
models = [f"{d}/weights.pk" for d in os.listdir("./") if "results_" in d]

test_data = ProtDataset(TEST_PATH, categories, CACHE_PATH)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=pad_batch)

# run predictions for each model, and get averaged prediction
pred_avg = tr.zeros((len(test_data), len(categories)))
for k, model in enumerate(models):
    print("load weights from", model)
    net = TLProtCNN(len(categories), device=DEVICE) 
    net.load_state_dict(tr.load(model))
    _, test_errate, pred, ref, _ = net.pred(test_loader)
    
    # k-ensemble score
    pred_avg += pred
    pred_avg_bin = tr.argmax(pred_avg, dim=1)    
    ensemble_errate = 1 - accuracy_score(ref, pred_avg_bin)
        
    msg = f"Model-{k+1:02d} error: {test_errate:.2f}"
    if k>0:
        msg += f", {k+1}-ensemble error: {ensemble_errate:.2f}"
    print(msg)
