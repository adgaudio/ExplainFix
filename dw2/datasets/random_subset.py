import torch as T


class RandomSubset(T.utils.data.Dataset):
    """
    Wraps a dataset and lies about its number of samples, n.

    Useful when paired with a dataloader that only iterates through n samples.
    In this case, one epoch evaluates only n samples.  The samples are randomly
    sampled with replacement from the wrapped dataset.
    """
    def __init__(self, dset: T.utils.data.Dataset, n: int):
        self.dset = dset
        self.n = n

    def __getitem__(self, i):
        return self.dset[T.randint(0, len(self.dset), (1,))[0]]

    def __len__(self):
        return self.n
