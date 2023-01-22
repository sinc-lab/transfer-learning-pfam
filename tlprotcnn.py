import torch as tr
from torch import nn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import numpy as np
from tqdm import tqdm
import os
from protcnn import ProtCNN

class TLProtCNN(nn.Module):
    def __init__(self, nclasses, lr=1e-3, nfilters=128, device="cuda", 
    logger=None, emb_size=1280):
        super().__init__()

        self.emb_size = emb_size
        self.logger = logger
        self.train_steps = 0
        self.dev_steps = 0
        
        self.device = device
        
        # use protcnn architecture
        self.cnn = ProtCNN(self.emb_size)
        self.fc = nn.Linear(1100, nclasses)
        
        self.loss = nn.CrossEntropyLoss()
        self.optim = tr.optim.Adam(self.parameters(), lr=lr)   

        self.to(device)
        self.device = device

    def forward(self, emb):
        """emb is the embedded sequence batch with shape [N, EMBSIZE, L]"""
        y = self.cnn(emb.to(self.device))
        y = self.fc(y.squeeze(2))    

        return y
    

    def fit(self, dataloader):
        
        avg_loss = 0
        self.train()
        self.optim.zero_grad()
        pred, ref = [], []
                
        for emb, y, _ in tqdm(dataloader):
            yhat = self(emb.to(self.device))
            y = y.to(self.device)
            
            loss = self.loss(yhat, y)
            loss.backward()
            avg_loss += loss.item()
            self.optim.step()
            self.optim.zero_grad()
            
            #pred.append(yhat.detach().cpu())
            #ref.append(y.cpu())
        
            if self.logger is not None:
                self.logger.add_scalar("Loss/train", loss, self.train_steps)
            self.train_steps+=1

        #pred = tr.cat(pred)
        #pred_bin = tr.argmax(pred, dim=1)    
        #ref = tr.cat(ref)    

        avg_loss /= len(dataloader)
        #acc = accuracy_score(ref, pred_bin)

        return avg_loss

    def pred(self, dataloader):
        test_loss = 0
        pred, ref, names = [], [], []
        self.eval() 

        for emb, y, name in tqdm(dataloader):
            with tr.no_grad():
                yhat = self(emb.to(self.device))
                y = y.to(self.device)
                test_loss += self.loss(yhat, y).item()
                
            names += name
            pred.append(yhat.detach().cpu())
            ref.append(y.cpu())
            
        pred = tr.cat(pred)
        pred_bin = tr.argmax(pred, dim=1)    
        ref = tr.cat(ref)    

        self.dev_steps += 1            
        test_loss /= len(dataloader)
        acc = accuracy_score(ref, pred_bin)
        if self.logger is not None:
            self.logger.add_scalar("Loss/dev", test_loss, self.dev_steps)
            balacc = balanced_accuracy_score(ref, pred_bin)
            self.logger.add_scalar("Error rate/dev", 1-acc, self.dev_steps)
            self.logger.add_scalar("Balanced acc/dev", balacc, self.dev_steps)
            
        return test_loss, 1-acc, pred, ref, names
