import torch
from torchvision import datasets, transforms
import ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

SUBSAMPLE_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PRETRAIN_SUBSAMPLE_SIZES = [1, 2, 4, 8, 16, 32, 64]
FINETUNE_SUBSAMPLE_SIZE = 256
for s in PRETRAIN_SUBSAMPLE_SIZES:
    assert s in SUBSAMPLE_SIZES
assert FINETUNE_SUBSAMPLE_SIZE in SUBSAMPLE_SIZES
NUM_CLASSES_DICT = {
    'mnist': 10,
    'emnist': 27
}

MNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3015,))
])
EMNIST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomRotation((90, 90)),
    transforms.Normalize((0.1722,), (0.3242,))
])

def get_subsampled_dataset(dataset, n_classes, k):
    '''
    Return a copy of the dataset that has k samples for each class
    '''
    all_inds = []
    for target in range(n_classes):
        indices = np.where(dataset.targets == target)[0]
        if len(indices) == 0:
            continue
        subsample = np.random.choice(indices, size=k)
        all_inds.append(subsample)
    all_inds = np.concatenate(all_inds)
    dataset = torch.utils.data.Subset(dataset, all_inds)
    return dataset


def get_dataset(dataset_str):
    '''
    Accepts dataset_str string as input. Must be one of ['mnist', 'emnist'].
    Returns:
        - A dictionary of subsampled versions of the training dataset. 
        - Test set. Go to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for documentation on pytorch datasets.

    Consider the following example code for how to use it:

    >>> subsampled_datasets, test_data = get_dataset('mnist')
    >>> small_train_set = subsampled_datasets[4]

    Then small_train_set will have 4 samples from every class of MNIST.
    '''
    if dataset_str not in NUM_CLASSES_DICT.keys():
        raise ValueError('Dataset name {} not recognized. Must be one of {}'.format(dataset_str, '[mnist, emnist]'))

    if dataset_str == 'mnist':
        train_data = datasets.MNIST('./data', train=True, transform=MNIST_TRANSFORM, download=True)
        test_data = datasets.MNIST('./data', train=False, transform=MNIST_TRANSFORM)

    else:
        assert dataset_str == 'emnist'
        train_data = datasets.EMNIST('./data', train=True, split='letters', transform=EMNIST_TRANSFORM, download=True)
        test_data = datasets.EMNIST('./data', train=False, split='letters', transform=EMNIST_TRANSFORM)

    n_classes = NUM_CLASSES_DICT[dataset_str]
    subsampled_datasets = {
        subsample_size: get_subsampled_dataset(train_data, n_classes, subsample_size) 
                        for subsample_size in SUBSAMPLE_SIZES
    }
    return subsampled_datasets, test_data

def get_new_data_loader(dataset, batch_size=8, drop_last=True):
    '''
    Torch datasets can be turned into python generators using torch.utils.data.DataLoader
    '''
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last
    )
    return iter(loader)

def test_data_balance(subsample_dict, dataset_str):
    # Assert that we get a balanced subsampling of classes
    n_classes = NUM_CLASSES_DICT[dataset_str]
    for subsample_size in subsample_dict.keys():
        example_loader = get_new_data_loader(subsample_dict[subsample_size], batch_size=n_classes, drop_last=False)
        labels = np.zeros(n_classes)
        for i, (data, target) in enumerate(example_loader):
            for t in target:
                labels[t] += 1
        for i in range(n_classes):
            if dataset_str == 'emnist' and i == 0:
                # EMNIST's class labels start at 1 for some reason, so we should have no elements of class 0
                assert labels[i] == 0
            elif labels[i] != subsample_size:
                print(labels)
                raise ValueError('Number of labels for class {} is {}; not equal to subsample_size {}'.format(i, labels[i], subsample_size))
