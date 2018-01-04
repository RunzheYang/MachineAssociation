import argparse
import torch
from torchvision import datasets, transforms
import visdom

parser = argparse.ArgumentParser(description='Machine Association - Vis MNIST')

parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                    help='batch size')

args = parser.parse_args()

data_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=args.batch_size, shuffle=True)

vis = visdom.Visdom()
vis.images(iter(data_loader).next()[0])