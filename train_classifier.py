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

parser.add_argument('--epochs', type=int, default=15, metavar='EPN',
                    help='number of epochs to train')
parser.add_argument('--optimizer', default='Adam', metavar='OPT',
                    help='optimizer: Adam | RMSprop')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')
parser.add_argument('--batch-size', type=int, default=50, metavar='BS',
                    help='input batch size for training')
parser.add_argument('--test-batch', type=int, default=1000, metavar='TB',
                    help='input batch size for testing')
parser.add_argument('--val-size', type=int, default=10000, metavar='VS',
                    help='size of validation set')

parser.add_argument('--save', default='classifier/saved/', metavar='SAVE',
                    help='path for saving trained classifiers')
parser.add_argument('--log', default='classifier/logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--name', default='anonymous', metavar='name',
                    help='specify a name for saving the model')

args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

# initialize model
model = get_classifier(args.classifier)
if use_cuda: model.cuda()

# get training, validation, and test dataset

train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
train_num = len(train_set) - args.val_size

train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size, sampler=chunk.Chunk(train_num, 0))

val_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=args.test_batch,
                sampler=chunk.Chunk(args.val_size, train_num))

test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=args.test_batch, shuffle=True)


# configure optimizer
optimizer = None
if args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'RMSprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)

# initialize monitor and logger
plotter = monitor.Plotter(args.name)
logger = monitor.Logger(args.log, args.name)

# train
model.train()
cnt = 0
for epoch in range(args.epochs):
    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        plotter.update_loss(cnt, loss.data[0])

        if cnt % 1000 == 0:
            prediction = output.data.max(1)[1]
            train_acc = prediction.eq(target.data).cpu().sum()/args.batch_size*100

            val_acc = 0.0
            for val_data, val_target in val_loader:
                if use_cuda:
                    val_data, val_target = val_data.cuda(), val_target.cuda()
                val_data, val_target = Variable(val_data, volatile=True), Variable(val_target)
                output = model(val_data, dropout=False)
                prediction = output.data.max(1)[1]
                val_acc += prediction.eq(val_target.data).cpu().sum()
            val_acc = 100. * val_acc / len(test_loader.dataset)

            plotter.update_acc(cnt, train_acc, val_acc)
            logger.update(cnt, loss, train_acc, val_acc)

            print('Train Step: {}\tLoss: {:.3f}\tTrain Acc: {:.3f}\tVal Acc: {:.3f}'.format(
                        cnt, loss.data[0], train_acc, val_acc))
        cnt += 1

# test
model.eval()
correct = 0
for data, target in test_loader:
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    data, target = Variable(data, volatile=True), Variable(target)
    output = model(data)
    prediction = output.data.max(1)[1]
    correct += prediction.eq(target.data).cpu().sum()

print('\nTest Accuracy: {:.2f}%'.format(100. * correct / len(test_loader.dataset)))

# save the final classifier
torch.save(model, "{}{}.pkl".format(args.save, args.name))
