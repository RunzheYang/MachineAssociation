import argparse
import visdom
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

parser.add_argument('--batch-size', type=int, default=100, metavar='BS',
                    help='input batch size for training')
parser.add_argument('--entropy-threshold', type=float, default=0.3, metavar="ET",
                    help='entropy threshold for selecting apprentice data')

parser.add_argument('--save', default='classifier/saved/', metavar='SAVE',
                    help='path for saving trained classifiers')
parser.add_argument('--name', default='mnist_lenet', metavar='name',
                    help='specify a name for saving the model')

args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

# initialize classifier
classifier = torch.load("{}{}.pkl".format(args.save, args.name))
if use_cuda:
    classifier.cuda()
    FloatTensor = torch.cuda.FloatTensor

# get training, validation, and test dataset

train_set = datasets.MNIST('./data', train=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./data', train=False, transform=transforms.ToTensor())


train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size, shuffle=True)

# test
classifier.eval()
vis = visdom.Visdom()
FloatTensor = torch.FloatTensor
apprentice_dateset = None
apprentice_data, apprentice_target = None, None

for data, target in train_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = classifier(Variable(data, volatile=True)).data
    entropy = (-torch.exp(output)*output).sum(dim=1)
    target_mask = entropy.gt(args.entropy_threshold)
    data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
    data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
    target = torch.masked_select(target, target_mask).cpu()
    if data.dim() == 0:
        continue
    vis.images(data)
    if apprentice_data is None:
        apprentice_data = data
    else:
        apprentice_data = torch.cat((apprentice_data, data), dim=0)
    if apprentice_target is None:
        apprentice_target = target
    else:
        apprentice_target = torch.cat((apprentice_target, target), dim=0)

for data, target in test_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    output = classifier(Variable(data, volatile=True)).data
    entropy = (-torch.exp(output)*output).sum(dim=1)
    target_mask = entropy.gt(args.entropy_threshold)
    data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
    data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
    target = torch.masked_select(target, target_mask).cpu()
    if data.dim() == 0:
        continue
    vis.images(data)
    if apprentice_data is None:
        apprentice_data = data
    else:
        apprentice_data = torch.cat((apprentice_data, data), dim=0)
    if apprentice_target is None:
        apprentice_target = target
    else:
        apprentice_target = torch.cat((apprentice_target, target), dim=0)

print(apprentice_data.size(), apprentice_target.size())

apprentice_dateset = (apprentice_data, apprentice_target)
with open('./data/mnist_apprentice.data', 'wb') as f:
    torch.save(apprentice_dateset, f)
