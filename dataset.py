from torch.utils.data import Dataset
import torch as tr 
import pickle 
from torch.utils.data  import Sampler
from random import shuffle 

from utils import read_original_data

class ProtDataset(Dataset):

    def __init__(self, data_path, categories, cache_path="emb_cache/", device="cuda"):
        
        self.device = device
        self.data = read_original_data(data_path)
        
        self.data["len"] = self.data.sequence.str.len()
        self.data = self.data.sort_values(by="len", ascending=False)
        self.categories = categories
        self.cache_path = cache_path

    def __getitem__(self, item):

        seq_name = self.data.iloc[item].sequence_name
        
        cache_file = f"{self.cache_path}{seq_name.replace('/', '-')}.pk"
        
        try:
            emb, label = pickle.load(open(cache_file, "rb"))
            emb = emb.float()
            label = tr.tensor(self.categories.index(label), dtype=tr.long)
        except FileNotFoundError:
            print(f"Error: {cache_file} is missing. Compute the embeddings first with compute_embeddings.py")
            exit()
        return emb, label, seq_name

    def __len__(self):
        return len(self.data)

    def get_lengths(self):
        return [l if l<1024 else 1024 for l in self.data.len]
    
def pad_batch(batch):
    """batch is a list of (embedding, label), with embedding with shape [1, E, L], L is variable"""
    max_len = max([b[0].shape[2] for b in batch] + [9]) # min len is 9
    emb = tr.zeros((len(batch),  batch[0][0].shape[1], max_len))
    labels = tr.zeros(len(batch), dtype=tr.long)
    names = []
    for k in range(len(batch)):
        emb[k, :, :batch[k][0].shape[2]] = batch[k][0]
        labels[k] = batch[k][1]
        names.append(batch[k][2])
    return emb, labels, names

class BatchSampler(Sampler):
    def __init__(self, seq_lengths, sorted=True, batch_size=32):
        """Sequences are sorted to minimize padding, then batches are shuffled. """
        self.batch_size = batch_size
        self.lengths = seq_lengths
        self.batched_ind = list(self.batch_indices())
        self.sorted = sorted
   
    def batch_indices(self):
        """generates batches sorted by length"""
        k = 0
        while k < len(self.lengths):
            batch = [p for p in range(k, min(k+self.batch_size, len(self.lengths)))]
            k += self.batch_size
            yield batch
            
    def __iter__(self):
        if not self.sorted:
            shuffle(self.batched_ind)
        return iter(self.batched_ind)
            
    def __len__(self):
        return len(self.batched_ind)
