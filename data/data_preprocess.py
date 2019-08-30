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
        root='./data/cifa10', train=True, download=True,
        transform=cifar_transform(is_training=True))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=20)

    eval_dataset = torchvision.datasets.CIFAR10(
        root='./data/cifa10', train=False, download=True,
        transform=cifar_transform(is_training=False))

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=20)

    len_eval_dataset = len(eval_dataset)
    return train_loader, eval_loader, len_eval_dataset


def svhn_transform(is_training=True):
    if is_training:
        # Former author's code settings
        transform_list = [transforms.Resize((40, 40)),
                          # brightness=30,
                          # transforms.ColorJitter(brightness=0., contrast=(0.5, 1.5)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]
    else:
        transform_list = [transforms.Resize((40, 40)),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])]

    transform_list = transforms.Compose(transform_list)
    return transform_list


def svhn_dataset():
    # Loading and normalizing SVHN
    train_dataset = torchvision.datasets.SVHN(
        root='./data/SVHN', split='train',
        transform=svhn_transform(is_training=True),
        download=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=20,
        drop_last=True)

    eval_dataset = torchvision.datasets.SVHN(
        root='./data/SVHN', split='test',
        transform=svhn_transform(is_training=False),
        download=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=20,
        drop_last=True)

    len_train_dataset = 572 * 128
    len_eval_dataset = 203 * 128
    return train_loader, eval_loader, len_train_dataset, len_eval_dataset
