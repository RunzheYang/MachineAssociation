import argparse
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
from utils.apprentice_loader import mnist
import visdom

parser = argparse.ArgumentParser(description='Machine Association - Vis MNIST')

parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='batch size')

args = parser.parse_args()

data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)

vis = visdom.Visdom()
vis.images(iter(data_loader).next()[0], opts=dict(title='MNIST'))

dataset = mnist.IDEAL('./data', 'times')
print('times', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='Times'))

dataset = mnist.IDEAL('./data', 'bradly')
print('bradly', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='Bradly Hand'))

dataset = mnist.IDEAL('./data', 'brush')
print('brush', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='Brush Script'))

dataset = mnist.IDEAL('./data', 'hannotate')
print('hannotate', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='Hannotate'))

dataset = mnist.IDEAL('./data', 'typewriter')
print('typewriter', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='Typewriter'))

dataset = mnist.IDEAL('./data', 'mixall')
print('mixall', len(dataset))
ideal_loader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True)
vis.images(iter(ideal_loader).next()[0], opts=dict(title='mixall'))
