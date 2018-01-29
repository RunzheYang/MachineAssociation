import argparse
import visdom
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from PIL import Image as IM
import numpy as np
import random

from classifier import get_classifier
from utils import chunk, monitor

# settings
parser = argparse.ArgumentParser(description='Train the classifier')
parser.add_argument('--font', default='times', metavar='FONT',
                    help='font type: times | ...')

parser.add_argument('--augment', default=False, action="store_true",
                    help='augment data for refiner training / testing')


args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

vis = visdom.Visdom()

# get type fonts
for i in range(10):
    im = IM.open("data/{}-digits".format(args.font))
    im.resize((28, 28), IM.ANTIALIAS)
    imgs.append(im)
    targets.append(i)
vis.images(imgs)


# # test
# classifier.eval()
# vis = visdom.Visdom()
# FloatTensor = torch.FloatTensor
# expert_dateset = None
# expert_data, expert_target = None, None
# test_expert_data, test_expert_target = None, None
# apprentice_dateset = None
# apprentice_data, apprentice_target = None, None
# test_apprentice_data, test_apprentice_target = None, None

# for data, target in train_loader:
#     if use_cuda:
#         data, target = data.cuda(), target.cuda()
#     output = classifier(Variable(data, volatile=True)).data
#     entropy = (-torch.exp(output) * output).sum(dim=1)
#     target_mask = entropy.le(args.entropy_threshold)
#     data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
#     data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
#     target = torch.masked_select(target, target_mask).cpu()
#     if data.dim() == 0:
#         continue
#     # vis.images(data)
#     if expert_data is None:
#         expert_data = data
#     else:
#         expert_data = torch.cat((expert_data, data), dim=0)
#     if expert_target is None:
#         expert_target = target
#     else:
#         expert_target = torch.cat((expert_target, target), dim=0)

# # if args.augment:
# #     expert_data = expert_data.repeat(2, 1, 1, 1)
# #     expert_target = expert_target.repeat(2)
# #     expert_data += torch.randn(expert_data.size())*0.001
# #     expert_data = torch.clamp(expert_data, min=0.0, max=1.0)

# print('There are {} instances in the expert dataset (Training)'.format(expert_data.size(0)))

# for data, target in test_loader:
#     if use_cuda:
#         data, target = data.cuda(), target.cuda()
#     output = classifier(Variable(data, volatile=True)).data
#     entropy = (-torch.exp(output) * output).sum(dim=1)
#     target_mask = entropy.le(args.entropy_threshold)
#     data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
#     data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
#     target = torch.masked_select(target, target_mask).cpu()
#     if data.dim() == 0:
#         continue
#     # vis.images(data)
#     if test_expert_data is None:
#         test_expert_data = data
#     else:
#         test_expert_data = torch.cat((test_expert_data, data), dim=0)
#     if test_expert_target is None:
#         test_expert_target = target
#     else:
#         test_expert_target = torch.cat((test_expert_target, target), dim=0)

# # if args.augment:
# #     test_expert_data = test_expert_data.repeat(2, 1, 1, 1)
# #     test_expert_target = test_expert_target.repeat(2)
# #     test_expert_data += torch.randn(test_expert_data.size())*0.001
# #     test_expert_data = torch.clamp(test_expert_data, min=0.0, max=1.0)

# print('There are {} instances in the expert dataset (Training)'.format(test_expert_data.size(0)))

# expert_data = torch.cat((expert_data, test_expert_data), dim=0)
# expert_target = torch.cat((expert_target, test_expert_target), dim=0)

# print('There are {} instances in the expert dataset (Total)'.format(expert_data.size(0)))

# expert_dateset = (expert_data, expert_target)
# with open('./data/mnist_expert_split.data', 'wb') as f:
#     torch.save(expert_dateset, f)

# for data, target in train_loader:
#     if use_cuda:
#         data, target = data.cuda(), target.cuda()
#     output = classifier(Variable(data, volatile=True)).data
#     entropy = (-torch.exp(output) * output).sum(dim=1)
#     target_mask = entropy.gt(args.entropy_threshold)
#     data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
#     data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
#     target = torch.masked_select(target, target_mask).cpu()
#     if data.dim() == 0:
#         continue
#     vis.images(data)
#     if apprentice_data is None:
#         apprentice_data = data
#     else:
#         apprentice_data = torch.cat((apprentice_data, data), dim=0)
#     if apprentice_target is None:
#         apprentice_target = target
#     else:
#         apprentice_target = torch.cat((apprentice_target, target), dim=0)

# if args.augment:
#     apprentice_data = apprentice_data.repeat(2, 1, 1, 1)
#     apprentice_target = apprentice_target.repeat(2)
#     apprentice_data += torch.randn(apprentice_data.size())*0.001
#     apprentice_data = torch.clamp(apprentice_data, min=0.0, max=1.0)

# print('There are {} instances in the apprentice dataset (Training)'.format(apprentice_data.size(0)))

# for data, target in test_loader:
#     if use_cuda:
#         data, target = data.cuda(), target.cuda()
#     output = classifier(Variable(data, volatile=True)).data
#     entropy = (-torch.exp(output) * output).sum(dim=1)
#     target_mask = entropy.gt(args.entropy_threshold)
#     data_mask = target_mask.view(-1, 1, 1, 1).expand(data.size())
#     data = torch.masked_select(data, data_mask).view(-1, data.size(1), data.size(2), data.size(3)).cpu()
#     target = torch.masked_select(target, target_mask).cpu()
#     if data.dim() == 0:
#         continue
#     vis.images(data)
#     if test_apprentice_data is None:
#         test_apprentice_data = data
#     else:
#         test_apprentice_data = torch.cat((test_apprentice_data, data), dim=0)
#     if test_apprentice_target is None:
#         test_apprentice_target = target
#     else:
#         test_apprentice_target = torch.cat((test_apprentice_target, target), dim=0)

# if args.augment:
#     test_apprentice_data = test_apprentice_data.repeat(2, 1, 1, 1)
#     test_apprentice_target = test_apprentice_target.repeat(2)
#     test_apprentice_data += torch.randn(test_apprentice_data.size())*0.001
#     test_apprentice_data = torch.clamp(test_apprentice_data, min=0.0, max=1.0)

# print('There are {} instances in the apprentice dataset (Training)'.format(test_apprentice_data.size(0)))

# apprentice_data = torch.cat((apprentice_data, test_apprentice_data), dim=0)
# apprentice_target = torch.cat((apprentice_target, test_apprentice_target), dim=0)

# print('There are {} instances in the apprentice dataset (Total)'.format(apprentice_data.size(0)))

# apprentice_dateset = (apprentice_data, apprentice_target)
# with open('./data/mnist_apprentice_split.data', 'wb') as f:
#     torch.save(apprentice_dateset, f)
