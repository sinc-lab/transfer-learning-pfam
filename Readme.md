# Transfer learning to annotate the protein universe

This is the source code for "Transfer learning: the key to functionally annotate the protein universe, 
L. A. Bugnon, E. Fenoy, A. Edera, J. Raad, G. Stegmayer* and D.H. Milone*

Research Institute for Signals, Systems and Computational Intelligence, sinc(i)
*{gstegmayer, dmilone}@sinc.unl.edu.ar

## Installation

We recomend using a python virtual environment such as conda or venv. 

Install pytorch (it may depend on your hardware setup, follow the recomendations in https://pytorch.org/get-started/locally/)

    pip install torch

Install the required packages

    pip install -r requirements.txt

## Run a demo version

As embedding sequences take some time, to get faster results sequence embeddings are precomputed.
You can download the embeddings of development and test partitions from 
https://drive.google.com/drive/folders/1M3tDoyA04k6Ucnhq3b-0MLJdAuhg4LZX
Copy and unzip the content to /data folder.

Then, a fast converging model can be trained using the dev partition and evaluating on the test partition.

    python train.py

## Reproducing results
To reproduce the results on the same data used in the paper, compute the embeddings for train sequencs with the following

    python compute_embeddings.py -i data/Clustered_data/train/

You can obtain the dev and test embeddings in the same way

Then, train 5 models with the full dataset

    python train.py --train data/Clustered_data/train/ -n 5

For reference, one model take about a day to converge using a RTX A5000 GPU.

To get the results of model ensembles, run 

    python summary.py
