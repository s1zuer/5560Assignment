import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10，按需也可切 MNIST
def get_data_loader(data_dir: str = "./data",
                    batch_size: int = 64,
                    train: bool = True,
                    num_workers: int = 2):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    if train:
        tfms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        tfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    dataset = datasets.CIFAR10(root=data_dir, train=train, download=True, transform=tfms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                        num_workers=0, pin_memory=False)
    return loader