import torch as T


def compose(*funcs):
    def _compose(*inpt):
        for fn in funcs:
            inpt = fn(*inpt)
        return inpt
    return _compose


class Preprocess(T.utils.data.Dataset):
    """Wrap a Dataset and add preprocessing:

        >>> dset = Preprocess(
            Dataset(...),
            lambda xy: T.tensor(xy[0]), T.tensor(xy[1]))

        Transform the dataset samples, before they are passed to a model or DataLoader
        """
    def __init__(self, kls, *fns):
        self.kls = kls
        self.fn = compose(fns) if len(fns) > 1 else fns[0]
    def __getitem__(self, x):
        return self.fn(self.kls[x])
    def __len__(self):
        return len(self.kls)




