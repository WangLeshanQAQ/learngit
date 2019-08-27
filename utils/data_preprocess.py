import torch
import torchvision
import torchvision.transforms as transforms


def cifar_transform(is_training=True):
    if is_training:
        transform_list = [transforms.RandomHorizontalFlip(),
                          transforms.Pad(padding=4, padding_mode='reflect'),
                          transforms.RandomCrop(32, padding=0),
                          transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]
    else:
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ]

    transform_list = transforms.Compose(transform_list)
    return transform_list


def cifar_dataset():
    # Loading and normalizing CIFAR10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=cifar_transform(is_training=True))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=30)

    eval_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=cifar_transform(is_training=True))

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=30)

    len_eval_dataset = len(eval_dataset)
    return train_loader, eval_loader, len_eval_dataset
