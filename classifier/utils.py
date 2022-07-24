import torch
import numpy as np
import random
import logging
import os
import torch.nn as nn

from typing import Dict, List, Tuple, Set
from models.mobilenetv3 import MobileNetV3
from models.resnet import ResNet
from models.vit import Vit
from torch.utils.tensorboard import SummaryWriter


def add_metrics_to_tensorboard(writer: SummaryWriter, metrics: Dict, epoch: int, mode: str, target: str) -> None:
    """
    Add metrics to Tensorboard logs

    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard log writer
    metrics : Dict
        Metrics value
    epoch : int
        Number of epoch
    mode : str
        Mode valid or train
    target : str
        Target name: gesture or leading_hand
    """
    logging.info(f'{mode}: metrics for {target}')
    logging.info(metrics)
    for key, value in metrics.items():
        writer.add_scalar(f'{key}_{target}/{mode}', value, epoch)


def add_params_to_tensorboard(writer: SummaryWriter, params: Dict, epoch: int, obj: str, not_logging: Set) -> None:
    """
    Add optimizer params to Tensorboard logs

    Parameters
    ----------
    writer : SummaryWriter
        Tensorboard log writer
    params : Dict
        Optimizer params for logging
    epoch : int
        Number of epoch
    obj : str
        Optimizer or learning scheduler for params logging
    not_logging : List
        Parameters that should not be logged
    """
    for param, value in params.items():
        if param not in not_logging:
            writer.add_scalar(f'{obj}/{param}', value, epoch)


def set_random_state(random_seed: int) -> None:
    """
    Set random seed for torch, numpy, random

    Parameters
    ----------
    random_seed: int
        Random seed from config
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_checkpoint(
        output_dir: str,
        config_dict: Dict,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        name: str
) -> None:
    """
    Save checkpoint dictionary

    Parameters
    ----------
    output_dir : str
        Path to directory model checkpoint
    config_dict : Dict
        Config dictionary
    model : nn.Module
        Model for checkpoint save
    optimizer : torch.optim.Optimizer
        Optimizer
    epoch : int
        Epoch number
    name : str
        Model name
    """
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir), exist_ok=True)

    checkpoint_path = os.path.join(output_dir, f'{name}.pth')

    checkpoint_dict = {
        'state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'config': config_dict
    }
    torch.save(checkpoint_dict, checkpoint_path)

# perceptron-posse

def build_model(
        model_name: str,
        num_classes: int,
        device: str,
        checkpoint: str = None,
) -> nn.Module:
    """
    Build modela and load checkpoint

    Parameters
    ----------
    model_name : str
        Model name e.g. ResNet18, MobileNetV3_small, Vitb32
    num_classes : int
        Num classes for each task
    checkpoint : str
        Path to model checkpoint
    device : str
        Cpu or CUDA device
    """
    models = {
        'ResNet18': ResNet(
            num_classes=num_classes,
            restype='ResNet18',
        ),
        'ResNet10': ResNet(
            num_classes=num_classes,
            restype='ResNet10',
        ),
        'ResNet20': ResNet(
            num_classes=num_classes,
            restype='ResNet20',
        ),
    }

    model = models[model_name]
    print(f' ---------- Chosen Model: {model_name} ---------- ')

    if checkpoint is not None:
        checkpoint = os.path.expanduser(checkpoint)
        if os.path.exists(checkpoint):
            checkpoint = torch.load(checkpoint, map_location=torch.device(device))["state_dict"]
            model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    return model


def collate_fn(batch: List) -> Tuple:
    """
    Collate func for dataloader

    Parameters
    ----------
    batch : List
        Batch of data
    """
    return tuple(zip(*batch))
