import os,sys,torch
import os.path as osp
import numpy as np, torch.nn as nn 
import torch.nn.functional as F 
from torch import optim 
from unet import UNet 
from init_path import IMG_DIR,MASK_DIR


def train_net(net,):
    img_dir='data/input/train'
    mask_dir='data/input/mask'
    checkpoint_dir='models'


