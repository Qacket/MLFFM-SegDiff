from inspect import isfunction
import torch


def exists(x):
    return x is not None



def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d



def normalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    # Function definition to normalize input tensor along height and width
    # Input:
    #   - tensor: torch.Tensor of size  (*other, height, width)
    # Output:
    #   - normalized_tensor: torch.Tensor of size (*other, height, width),
    #                        which is the result of normalization
    """
    min_vals = torch.min(tensor, dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]

    max_vals = torch.max(tensor, dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

    normalized_tensor = (tensor - min_vals) / (max_vals - min_vals)
    return normalized_tensor



def mean_of_list_of_tensors(list_of_tensors):
    mean_x = torch.zeros_like(list_of_tensors[0])
    for t in list_of_tensors:
        mean_x += t
    mean_x /= len(list_of_tensors)
    return mean_x

