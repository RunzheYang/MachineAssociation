import argparse
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.apprentice_loader import mnist
from torchvision import transforms, datasets
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

parser.add_argument('--verbose', default=False, action='store_true',
                    help='print the labels and predictions')


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
# dataset = mnist.APPRENTICE('./data')
# dataset = mnist.EXPERT('./data')
# train_size = len(dataset) - args.test_size
#
# train_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=args.batch_size, sampler=chunk.Chunk(train_size, 0))
#
# test_loader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=args.test_batch,
#     sampler=chunk.Chunk(args.test_size, train_size))

train_set = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
train_size = len(train_set)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.test_batch, shuffle=True)

# visualize results
vis = visdom.Visdom()
classifier.eval()
refiner.eval()
if args.on_test:
    original_correct = 0
    refined_correct = 0
    original_entropy_sum = 0.0
    refined_entropy_sum = 0.0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        refined_data = refiner(data)
        original_output = classifier(data)
        refined_output = classifier(refined_data)
        original_entropy_sum += (-torch.exp(original_output) * original_output).data.cpu().sum()
        refined_entropy_sum += (-torch.exp(refined_output) * refined_output).data.cpu().sum()
        original_prediction = original_output.data.max(1)[1]
        refined_prediction = refined_output.data.max(1)[1]
        original_correct += original_prediction.eq(target.data).cpu().sum()
        refined_correct += refined_prediction.eq(target.data).cpu().sum()
    print('\nTest Accuracy: {:.2f}%'.format(100. * original_correct / args.test_size))
    print('\nRefined Test Accuracy: {:.2f}%'.format(100. * refined_correct / args.test_size))
    print('\nTest Entropy: {:.3f}'.format(original_entropy_sum / args.test_size))
    print('\nRefined Test Entropy: {:.3f}'.format(refined_entropy_sum / args.test_size))
else:
    original_correct = 0
    refined_correct = 0
    original_entropy_sum = 0.0
    refined_entropy_sum = 0.0
    for data, target in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        refined_data = refiner(data)
        original_output = classifier(data)
        refined_output = classifier(refined_data)
        original_entropy_sum += (-torch.exp(original_output) * original_output).data.cpu().sum()
        refined_entropy_sum += (-torch.exp(refined_output) * refined_output).data.cpu().sum()
        original_prediction = original_output.data.max(1)[1]
        refined_prediction = refined_output.data.max(1)[1]
        original_correct += original_prediction.eq(target.data).cpu().sum()
        refined_correct += refined_prediction.eq(target.data).cpu().sum()
    print('\n------------------\nRefiner: {}'.format(args.refiner_name))
    print('\nTrain Accuracy: {:.2f}%'.format(100. * original_correct / train_size))
    print('\nRefined Train Accuracy: {:.2f}%'.format(100. * refined_correct / train_size))
    print('\nTrain Entropy: {:.3f}'.format(original_entropy_sum / train_size))
    print('\nRefined Train Entropy: {:.3f}'.format(refined_entropy_sum / train_size))
