import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from get_data import \
        get_dataset, \
        get_new_data_loader, \
        test_data_balance, \
        SUBSAMPLE_SIZES, \
        FINETUNE_SUBSAMPLE_SIZE, \
        PRETRAIN_SUBSAMPLE_SIZES, \
        NUM_CLASSES_DICT
from network_training import train

class Net(nn.Module):
    """
    Neural network model that can pre-train on one image dataset and then finetune onto another image dataset.
    """
    def __init__(self, pre_train_classes, generalization_classes):
        super(Net, self).__init__()
        self.pre_train_classes = pre_train_classes
        self.generalization_classes = generalization_classes

        # Initialize neural network layers used
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.lin_size = 256

        self.pretrainer = nn.Linear(self.lin_size, self.pre_train_classes)
        self.generalizer = nn.Linear(self.lin_size, self.generalization_classes)

    def apply_convs(self, x):
        """
        Extract convolutional features from the input.

        Our network has two layers of feature extraction. Each layer consists of:
            - convolution on the features
            - 2x2 max_pool on the conv output (with stride 2)
            - relu activation

        Furthermore, you may apply dropout after the second layer's convolutions to encourage generalization.
        We recommend using torch.nn.functional (F.) for max pooling and relu
        """
        ### YOUR CODE HERE
        ### END CODE

        # Reshape the conv outputs so that we can apply linear layers to them
        x = x.view(-1, self.lin_size)

        return x
    
    def generalize(self, x):
        """
        Produce a prediction on x for finetuning.
        We first extract the convolutional features and then apply the generalization head onto those.
        """
        ### YOUR CODE HERE
        ### END CODE

        # Note: we use LOG SOFTMAX here, rather than just softmax.
        # This must be consistent with the cross_entropy implementation
        return F.log_softmax(x, dim=-1)
        
    def forward(self, x):
        """
        Produce a prediction on x for pretraining
        We first extract the convolutional features and then apply the pretraining head onto those.
        """
        ### YOUR CODE HERE
        ### END CODE

        # Note: we use LOG SOFTMAX here, rather than just softmax.
        # This must be consistent with the cross_entropy implementation
        return F.log_softmax(x, dim=-1)



def test_network(n_samples=1, n_batches=500, minimum_acc=0.9, maximum_loss=0.05, data_str='mnist'):
    """
    We test that the network can train correctly by letting it learn a small subsample of the MNIST dataset.
        - We default to 1 sample per class to make this a particularly easy task.
    We expect that, in very few batches, the network gets high accuracy and low loss.

    Feel free to run this method with n_samples > 1 to check how well your network works.
    """
    data_subsamples, test_dataset = get_dataset(data_str)
    try:
        toy_dataset = data_subsamples[n_samples]
    except KeyError:
        raise KeyError("There is no subsampled MNIST dataset with {} samples per class".format(n_samples))

    n_classes = NUM_CLASSES_DICT[data_str]
    network = Net(pre_train_classes=n_classes, generalization_classes=n_classes)
    optimizer = optim.SGD(network.parameters(), lr=0.01)

    train_losses, train_accuracies, _, _ = train(
        network,
        forward_call=network,
        optimizer=optimizer,
        train_dataset=toy_dataset,
        test_dataset=test_dataset, # we don't care about the test set when doing unit testing
        n_batches=n_batches,
        batch_size=8,
        n_classes=n_classes,
    )

    try:
        # train_losses and train_accuracies hold the losses and accuracies for every batch during training
        # We only need high accuracy and low loss on the last batches during training
        np.testing.assert_array_less(train_losses[-1], maximum_loss)
        np.testing.assert_array_less(minimum_acc, train_accuracies[-1])
    except AssertionError as E:
        raise E


if __name__ == '__main__':
    print('Due to randomness, the network tests may occassionally fail on correct implementations. If so, try to rerun a few times to see if the problem persists.')
    
    data_str = 'mnist'
    test_network(n_samples=1, n_batches=500, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)
    test_network(n_samples=4, n_batches=1000, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)
    test_network(n_samples=16, n_batches=2000, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)

    data_str = 'emnist'
    test_network(n_samples=1, n_batches=1000, minimum_acc=0.9, maximum_loss=0.5, data_str=data_str)
    test_network(n_samples=4, n_batches=2000, minimum_acc=0.9, maximum_loss=0.5, data_str=data_str)
    test_network(n_samples=16, n_batches=4000, minimum_acc=0.9, maximum_loss=1.0, data_str=data_str)

    print('All simple network training tests passed!')
