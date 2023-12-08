import torch
import tqdm
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from graph_transformers import Transformer, GraphDataset, AverageMeter
from matplotlib import pyplot as plt 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')