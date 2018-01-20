import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils.apprentice_loader import mnist
from torchvision import transforms
from torch.autograd import Variable

from refiner import get_refiner
from utils import chunk, monitor

# Training settings
parser = argparse.ArgumentParser(description='Train the classifier')
parser.add_argument('--refiner', default='unet', metavar='RFN',
                    help='refiner: unet')

parser.add_argument('--lbd', type=float, default=1.0, metavar='LAMBDA',
                    help='coefficient for balancing simplicity and effectiveness')

parser.add_argument('--epochs', type=int, default=100, metavar='EPN',
                    help='number of epochs to train')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size', type=int, default=50, metavar='BS',
                    help='input batch size for training')
parser.add_argument('--test-batch', type=int, default=100, metavar='TB',
                    help='input batch size for testing')
parser.add_argument('--test-size', type=int, default=200, metavar='VS',
                    help='size of validation set')

parser.add_argument('--classifier-path', default='classifier/saved/', metavar='CP',
                    help='path for used classifier')
parser.add_argument('--classifier-name', default='mnist_lenet', metavar='name',
                    help='specify the name of used classifier')

parser.add_argument('--delta', default='raw_l1', metavar='DELTA',
                    help='delta is the similarity distance: raw_l1 | feature_l1')
parser.add_argument('--eta', default='nll', metavar='ETA',
                    help='eta is the efficacy loss: nll | entropy')

parser.add_argument('--save', default='refiner/saved/', metavar='SAVE',
                    help='path for saving trained refiner')
parser.add_argument('--log', default='refiner/logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--name', default='mnist_unet', metavar='name',
                    help='specify a name for saving the model')

args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

# initialize classifier
classifier = torch.load("{}{}.pkl".format(args.classifier_path, args.classifier_name))
if use_cuda: classifier.cuda()

# initialize refiner
refiner = get_refiner(args.refiner)
if use_cuda: refiner.cuda()

# define similarity distance
if args.delta == "raw_l1":
    def delta(clf, rfd, d):
        return torch.norm(rfd-d, p=1)/rfd.view(-1,1).size(0)
elif args.delta == "feature_l1":
    def delta(clf, rfd, d):
            f1 = copy.deepcopy(clf)(rfd, extract_feature=True)
            f2 = copy.deepcopy(clf)(d, extract_feature=True)
            return torch.norm(f1 - f2, p=1)/f1.view(-1,1).size(0)
else:
    print("delta {} is not defined".format(args.delta))

# define efficacy loss
if args.eta == "nll":
    def eta(clf, rfd, tar):
        return F.nll_loss(clf(rfd), tar)
elif args.eta == 'entropy':
    def eta(clf, rfd, tar):
        output = clf(rfd)
        return (-torch.exp(output) * output).sum() / output.size(0)
else:
    print("eta {} is not defined".format(args.eta))

# get training, validation, and test dataset

dataset = mnist.APPRENTICE('./data')
train_size = len(dataset) - args.test_size

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, sampler=chunk.Chunk(train_size, 0))

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.test_batch,
    sampler=chunk.Chunk(args.test_size, train_size))

# configure optimizer
optimizer = None
if args.optimizer == 'Adam':
    optimizer = optim.Adam(refiner.parameters(), lr=args.lr)
elif args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop(refiner.parameters(), lr=args.lr)

# initialize monitor and logger
plotter = monitor.Plotter(args.name)
logger = monitor.Logger(args.log, args.name)

# train
classifier.eval()
refiner.train()
cnt = 0
for epoch in range(args.epochs):
    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        refined_data = refiner(data)
        # loss = similarity distance + efficacy loss
        loss = (1-args.lbd) * delta(classifier, refined_data, data) + args.lbd * eta(classifier, refined_data, target)
        plotter.update_loss(cnt, loss.data[0])
        loss.backward()
        optimizer.step()
        if cnt % 100 == 0:
            refined_data = refined_data.detach()
            refined_data.volatile=True
            output = classifier(refined_data)
            prediction = output.data.max(1)[1]
            train_acc = prediction.eq(target.data).cpu().sum() / prediction.size(0) * 100

            val_acc = 0.0
            for val_data, val_target in test_loader:
                if use_cuda:
                    val_data, val_target = val_data.cuda(), val_target.cuda()
                val_data, val_target = Variable(val_data, volatile=True), Variable(val_target)
                refined_data = refiner(val_data)
                output = classifier(refined_data)
                prediction = output.data.max(1)[1]
                val_acc += prediction.eq(val_target.data).cpu().sum()
            val_acc = 100. * val_acc / args.test_size

            plotter.update_acc(cnt, train_acc, val_acc)
            logger.update(cnt, loss, train_acc, val_acc)

            print('Train Step: {}\tLoss: {:.3f}\tTrain Acc: {:.3f}\tVal Acc: {:.3f}'.format(
                cnt, loss.data[0], train_acc, val_acc))
        cnt += 1
# test
classifier.eval()
refiner.eval()
correct = 0
for data, target in test_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    refined_data = refiner(data)
    output = classifier(refined_data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).cpu().sum()

print('\nTest Accuracy: {:.2f}%'.format(100. * correct / args.test_size))

# save the final classifier
torch.save(refiner, "{}{}.pkl".format(args.save, args.name))
