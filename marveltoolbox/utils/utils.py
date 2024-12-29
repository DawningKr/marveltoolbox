import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from marveltoolbox.LoRA import LoRA


def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if "model" in str(layer.__class__):
            continue
        if "container" in str(layer.__class__):
            continue
        else:
            if "batchnorm" in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def get_jacobian(fun, x, noutputs):
    x_size = list(x.size())
    x = (
        x.unsqueeze(0)
        .repeat(noutputs, *([1] * (len(x_size))))
        .detach()
        .requires_grad_(True)
    )
    y = fun(x)
    y.backward(torch.eye(noutputs))
    return x.grad.view(noutputs, *x_size)


def Hessian_matrix(fun, x):
    # y is a scalar Tensor
    def get_grad(xx):
        y = fun(xx)
        (grad,) = torch.autograd.grad(
            y, xx, create_graph=True, grad_outputs=torch.ones_like(y)
        )
        return grad

    x_size = x.numel()
    return get_jacobian(get_grad, x, x_size)


def analyze_latent_space(z, y, class_num=2):
    gaussians = []
    for c in range(class_num):
        idx = y.eq(c).nonzero().view(-1)
        n = len(idx)
        z_c = z[idx]
        mu_c = torch.mean(z_c, dim=0, keepdim=True)
        v_c = z_c - mu_c.repeat(n, 1)
        var_c = 1 / n * v_c.transpose(0, 1).matmul(v_c)
        gaussian = MultivariateNormal(
            loc=mu_c.clone().detach(), covariance_matrix=var_c.clone().detach()
        )
        gaussians.append(gaussian)
    return gaussians


def log_pz(z, y, gaussians, device):
    n, c = len(z), len(gaussians)
    V = torch.zeros(n, c, device=device)
    mask = F.one_hot(y, num_classes=c).type_as(z).to(device)
    for i in range(c):
        V[:, i] = gaussians[i].log_prob(z)
    A = V * mask
    A = torch.sum(A, dim=1)
    return A.view(n, -1)


def sample(n, netD, gaussians):
    # check which device netD is on
    device = next(netD.parameters()).device
    
    samples = []
    K = len(gaussians)
    for i in range(n):
        idx = np.random.choice(np.arange(K))
        samples.append(gaussians[idx].sample())
    sample_z = torch.cat(samples, dim=0).to(device)
    sample_x = netD(sample_z)
    return sample_x


def one_hot_encode(labels, n_labels):
    """
    Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    """

    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y


def logit(x):
    """
    Element-wise logit (inverse logistic sigmoid).
    :param x: numpy array
    :return: numpy array
    """
    return np.log(x / (1.0 - x))



"""
These two functions helps to load or save relevant checkpoint files
"""

def save_checkpoint(state_dict, is_best, file_path="./", flag=""):
    """
    Save model's state_dict to the given path
    Args:
        state_dict (dict): a dict which contains a model's parameters
        is_best (bool): whether the given state_dict is the best result
        file_path (str, optional): the path where the state_dict will be stored. Defaults to "./".
        flag (str, optional): a flag containing the model's relevant information. Defaults to "".
    """
    file_name = os.path.join(file_path, "checkpoint_{}.pth.tar".format(flag))
    torch.save(state_dict, file_name)
    if is_best:
        best_file_name = os.path.join(file_path, "model_best_{}.pth.tar".format(flag))
        shutil.copyfile(file_name, best_file_name)


def load_checkpoint(is_best, file_path="./", flag=""):
    checkpoint = None
    if is_best:
        checkpoint_file = os.path.join(file_path, "model_best_{}.pth.tar".format(flag))
    else:
        checkpoint_file = os.path.join(file_path, "checkpoint_{}.pth.tar".format(flag))

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file, lambda storage, loc: storage)
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_file))
    return checkpoint


def lora_state_dict(model):
    """
    Get a state_dict of the given model with lora parts only
    :param model: a model inherited torch.nn.Module and LoRA
    :return: dict
    """
    state_dict = model.state_dict()
    return {key: state_dict[key] for key in state_dict.keys() if "lora" in key}


def set_lora_configs_all(model: LoRA, rank: int, alpha: int, enable_lora=True):
    """
    Set lora configs for all lora layers in the given model using the same parameters
    """
    lora_layers = model.get_lora_layers()
    for layer in lora_layers:
        layer.set_lora_configs(rank, alpha)
        if enable_lora:
            layer.set_lora_status(True)
    return model
