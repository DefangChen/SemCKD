import torch
from torchvision.datasets import CIFAR100


def stratified_split(dataset, n_per_class):
    count = {}
    indices = torch.randperm(len(dataset))
    targets = dataset.targets
    train_indices = []
    val_indices = []
    for i in indices:
        t = targets[i]
        count[t] = count.get(t, 0) + 1
        if count[t] <= n_per_class:
            val_indices.append(i.item())
        else:
            train_indices.append(i.item())
    return train_indices, val_indices


if __name__ == '__main__':
    trainset = CIFAR100(root='data', train=True, download=True)
    for n_per_class in [50, 150, 250]:
        _, indices = stratified_split(trainset, n_per_class)
        rand_labels = torch.randint(100, (len(indices), ))
        noise = {
            x: rand_labels[i].item() for i, x in enumerate(indices)
        }
        torch.save(noise, 'dataset/cifar100_label_noise_%d_per_class.pth' % n_per_class)
