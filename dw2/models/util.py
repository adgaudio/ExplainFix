import torch as T
from typing import Union, Generator


def iter_conv2d(model: T.nn.Module, include_spatial:bool=True, include_1x1:bool=False,
                return_name:bool=False):
    for module_name, module in model.named_modules():
        if isinstance(module, T.nn.Conv2d):
            if module.kernel_size == (1,1):
                if not include_1x1:
                    continue
            elif not include_spatial:
                continue
            #  assert module.kernel_size[0] > 1
            if return_name:
                yield module_name, module
            else:
                yield module


def extract_all_spatial_filters(
        model:T.nn.Module, requires_grad=False,
        HW:Union[str,list[tuple[int,int]]]='all'
        ) -> Generator[list[tuple[int,int], int, T.Tensor], None, None]:
    for layer_idx, conv2d in enumerate(iter_conv2d(model)):
        o,i,h,w = conv2d.weight.shape
        F = conv2d.weight.reshape(o*i, h*w)
        if requires_grad is False:
            F = F.detach()
        if HW == 'all' or (h,w) in HW:
            yield ( (h,w), layer_idx, F )


def extract_spatial_filters(model:T.nn.Module, H:int, W:int, requires_grad=False, as_matrix=True) -> T.Tensor:
    F = []
    for conv2d in iter_conv2d(model):
        o,i,h,w = conv2d.weight.shape
        if h == H and w == W:
            tmp = conv2d.weight.reshape(o*i, h*w)
            if requires_grad is False:
                tmp = tmp.detach()
            F.append(tmp)
    if as_matrix:
        if len(F):
            F = T.cat(F, 0)
        else:
            F = T.tensor([[]], device=conv2d.weight.device)
    return F
