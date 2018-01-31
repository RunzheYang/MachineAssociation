import argparse
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils.apprentice_loader import mnist

from PIL import Image as IM
from PIL import ImageChops
import numpy as np
import random

from classifier import get_classifier
from utils import chunk, monitor

# settings
parser = argparse.ArgumentParser(description='Train the classifier')
parser.add_argument('--font', default='times', metavar='FONT',
                    help='font type: times | bradly | brush | hannotate | typewriter | mixall')


args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

vis = visdom.Visdom()

imgs = []
targets = []

# get type fonts
if not args.font == "mixall":
    for angle in range(-20, 20+1, 1):
        for xoff in range(-2, 2+1, 1):
            for yoff in range(-2, 2+1, 1):
                for i in range(10):
                    im = IM.open("data/{}-digits/{}.png".format(args.font, i))
                    im = im.convert('L')
                    im = im.rotate(angle)
                    im = im.resize((28, 28), IM.ANTIALIAS)
                    im = ImageChops.offset(im, xoff, yoff)
                    imgs.append(np.array(im).reshape(1,28,28))
                    targets.append(i)
else:
    for angle in range(-20, 20+1, 1):
        for xoff in range(-2, 2+1, 1):
            for yoff in range(-2, 2+1, 1):
                for i in range(10):
                    for font in ["times", "bradly", "brush", "hannotate", "typewriter"]:
                        im = IM.open("data/{}-digits/{}.png".format(font, i))
                        im = im.convert('L')
                        im = im.rotate(angle)
                        im = im.resize((28, 28), IM.ANTIALIAS)
                        im = ImageChops.offset(im, xoff, yoff)
                        imgs.append(np.array(im).reshape(1,28,28))
                        targets.append(i)
imgs = np.array(imgs, dtype=np.float)/255.0
targets = np.array(targets)
permutation = np.random.permutation(len(imgs))
s_imgs = np.empty(imgs.shape, dtype=imgs.dtype)
s_targets = np.empty(targets.shape, dtype=targets.dtype)
for old_ind, new_ind in enumerate(permutation):
    s_imgs[new_ind] = imgs[old_ind]
    s_targets[new_ind] = targets[old_ind]
imgs = torch.from_numpy(s_imgs).type(torch.FloatTensor)
targets = torch.from_numpy(s_targets)
# vis.images(imgs[0:10])
# print(targets[0:10])

ideal_dateset = (imgs, targets)
with open('./data/ideal_{}.data'.format(args.font), 'wb') as f:
    torch.save(ideal_dateset, f)
