import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from classifier import get_classifier
from utils import chunk, monitor

# Training settings
parser = argparse.ArgumentParser(description='Train the classifier')
parser.add_argument('--classifier', default='lenet', metavar='CLF',
                    help='classifier: lenet')

parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for training')

parser.add_argument('--save', default='classifier/saved/', metavar='SAVE',
                    help='path for saving trained classifiers')
parser.add_argument('--name', default='mnist_lenet', metavar='name',
                    help='specify a name for saving the model')

args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

# initialize model
classifier = torch.load("{}{}.pkl".format(args.save, args.name))
if use_cuda: model.cuda()

# get training, validation, and test dataset

test_set = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())

test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.test_batch, shuffle=True)

# test
model.eval()
vis = visdom.Visdom()
data, target = iter(test_loader).next()
vis.images(data)
output = model(data)
print(output)


