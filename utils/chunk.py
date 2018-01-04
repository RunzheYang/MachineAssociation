import torch
from torch.utils.data import sampler

class Chunk(sampler.Sampler):
    """Samples elements randomly from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        # return iter(torch.randperm(self.num_samples).long() + self.start)
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples