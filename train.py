from datetime import datetime
import os
import torch as tr
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import argparse

from dataset import ProtDataset
from dataset import pad_batch, BatchSampler
from tlprotcnn import TLProtCNN

parser = argparse.ArgumentParser()
parser.add_argument("--train", default=f"data/Clustered_data/dev/")
parser.add_argument("--dev", default=f"data/Clustered_data/dev/")
parser.add_argument("--test", default=f"data/Clustered_data/test/")
parser.add_argument("--cache", default="data/")
parser.add_argument("-n", type=int, default=1)

args = parser.parse_args()

LR = 1e-3
DEVICE = "cuda"
NEPOCH = 1000
PATIENCE = 10
BATCH_SIZE = 32
WORKERS = 4

categories = [item.strip() for item in open("data/categories.txt")]

train_data = ProtDataset(args.train, categories, cache_path=args.cache)
    
train_loader = DataLoader(train_data, batch_sampler=BatchSampler(train_data.get_lengths(), sorted=False, batch_size=BATCH_SIZE), collate_fn=pad_batch, num_workers=WORKERS)

dev_data = ProtDataset(args.dev, categories, cache_path=args.cache)

dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, collate_fn=pad_batch, num_workers=WORKERS)

test_data = ProtDataset(args.test, categories, cache_path=args.cache)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=pad_batch, num_workers=WORKERS)

# Train N models
for nrepeat in range(args.n):
    OUT_DIR = f"results_{str(datetime.now())}/"
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    logger = SummaryWriter(OUT_DIR)

    counter, best_err = 0, 999
    net = TLProtCNN(len(categories), lr=LR, device=DEVICE, logger=logger)

    for epoch in range(NEPOCH):

        train_loss = net.fit(train_loader)
        dev_loss, dev_err, _, _, _ = net.pred(dev_loader)

        # early stop
        if dev_err < best_err:
            best_err = dev_err
            tr.save(net.state_dict(), f"{OUT_DIR}weights.pk")
            counter = 0
        else:
            counter += 1
            if counter > PATIENCE:
                break
        print(f"{epoch}: train loss {train_loss:.3f}, dev loss {dev_loss:.3f}, dev err {dev_err:.3f}")

    net = TLProtCNN(len(categories), lr=LR, device=DEVICE) 
    net.load_state_dict(tr.load(f"{OUT_DIR}weights.pk"))
    test_loss, test_errate, _, _, _ = net.pred(test_loader)
    print(f"test_loss {test_loss:.2f} test_errate {test_errate:.2f}")
