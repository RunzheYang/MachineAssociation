import argparse
import torch
from torch.autograd import Variable
from utils.apprentice_loader import mnist
from torchvision import transforms
from utils import chunk, monitor
import visdom

# Test settings
parser = argparse.ArgumentParser(description='Train the classifier')

parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                    help='input batch size for original training')
parser.add_argument('--test-batch', type=int, default=64, metavar='TB',
                    help='input batch size for testing')
parser.add_argument('--test-size', type=int, default=200, metavar='VS',
                    help='size of validation set')

parser.add_argument('--classifier-path', default='classifier/saved/', metavar='CP',
                    help='path for used classifier')
parser.add_argument('--classifier-name', default='mnist_lenet', metavar='name',
                    help='specify the name of used classifier')

parser.add_argument('--refiner-path', default='refiner/saved/', metavar='CP',
                    help='path for used refiner')
parser.add_argument('--refiner-name', default='mnist_unet', metavar='name',
                    help='specify the name of used refiner')

parser.add_argument('--on-test', default=False, action='store_true',
                    help='test refiner on the test set')


args = parser.parse_args()

# use cuda
use_cuda = torch.cuda.is_available()

# initialize classifier
classifier = torch.load("{}{}.pkl".format(args.classifier_path, args.classifier_name))
if use_cuda: classifier.cuda()

# initialize refiner
refiner = torch.load("{}{}.pkl".format(args.refiner_path, args.refiner_name))
if use_cuda: refiner.cuda()

# prepare dataset
dataset = mnist.APPRENTICE('./data')
train_size = len(dataset) - args.test_size

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size, sampler=chunk.Chunk(train_size, 0))

test_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.test_batch,
    sampler=chunk.Chunk(args.test_size, train_size))

# visualize results
vis = visdom.Visdom()
classifier.eval()
refiner.eval()
if args.on_test:
    data, target = iter(test_loader).next()
    plt_title = ('Apprentice Test Data - {}'.format(args.refiner_name),
                 'Refined Test Data - {}'.format(args.refiner_name))
else:
    data, target = iter(train_loader).next()
    plt_title = ('Apprentice Traning Data - {}'.format(args.refiner_name),
                 'Refined Training Data - {}'.format(args.refiner_name))

if use_cuda:
    data, target = data.cuda(), target.cuda()

target = Variable(target)

original_data = Variable(data, volatile=True)
original_output = classifier(original_data)
original_prediction = original_output.data.max(1)[1]
original_acc = original_prediction.eq(target.data).cpu().sum() / args.batch_size * 100

refined_data = refiner(Variable(data, volatile=True))
refined_output = classifier(refined_data)
refined_prediction = refined_output.data.max(1)[1]
refined_acc = refined_prediction.eq(target.data).cpu().sum() / args.batch_size * 100

print("original acc: {}%\nrefined acc: {}%\n".format(original_acc, refined_acc))

vis.images(original_data.data.cpu(),
           opts=dict(title=plt_title[0],
                     caption="original acc: {}%".format(original_acc)))
vis.images(refined_data.data.cpu(),
           opts=dict(title=plt_title[1],
                     caption="refined acc: {}%".format(refined_acc)))
