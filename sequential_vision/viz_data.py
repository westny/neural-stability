import os
import torch
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from lightning.pytorch import seed_everything


class PermuteTransform:
    def __init__(self):
        self.perm_idx = torch.randperm(28 * 28)

    def __call__(self, tensor):
        # Assuming tensor is of size (1, 28, 28) for grayscale MNIST images
        return tensor.view(-1)[self.perm_idx].view(1, 28, 28)


seed_everything(42)

transform_list = [
    transforms.ToTensor()
]
transform = transforms.Compose(transform_list)

dataset = 'pmnist'

root = os.path.join(os.getcwd(), '../data')

if dataset == 'mnist':
    test = datasets.MNIST(root, train=False, download=True, transform=transform)
elif dataset == 'pmnist':
    transform_list = [
        transforms.ToTensor(),
        PermuteTransform()]

    transform = transforms.Compose(transform_list)
    test = datasets.MNIST(root, train=False, download=True, transform=transform)

elif dataset == 'cifar10':
    test = datasets.CIFAR10(root, train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(test, batch_size=512, shuffle=True)
gen = iter(test_loader)
# plot 10 x 10 samples based on their label
data = {}
batch = next(gen)
inp, trg = batch
for i in range(10):
    data[i] = []
    idx = (trg == i).nonzero(as_tuple=True)[0]
    for j in range(10):
        data[i].append(inp[idx[j]])

nrows = len(data)
fig, axes = plt.subplots(nrows=nrows, ncols=10, figsize=(10, nrows))
for i, (label, inp) in enumerate(data.items()):
    for j in range(10):
        if dataset == 'mnist' or dataset == 'pmnist':
            axes[i, j].imshow(inp[j].view(28, 28), cmap='gray')
        elif dataset == 'cifar10':
            axes[i, j].imshow(inp[j].permute(1, 2, 0))
        axes[i, j].axis('off')
    # Set the label for the row
    axes[i, 0].set_ylabel(i)
plt.tight_layout()
plt.savefig(f'../img/{dataset}_samples.png', dpi=300)

plt.show()
