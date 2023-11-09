from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotting import plot_train_test

from get_data import get_new_data_loader
from utils import torch_onehot, cross_entropy
from augmentations import augment

def train_step(
    data,
    target,
    forward_call,
    optimizer,
    n_classes,
    semisupervised=False,
    batch_size=8
):
    '''
    Run a single step of training:
        - apply network to dataset
        - calculate loss
        - backpropagate
        - record accuracy (if applicable)

    Recall that if we are using the mixup or collage augmentations, our class labels are NOT onehot.
    Therefore, in this case the idea of 'accuracy' is poorly defined. As a result, we check whether we are in the
        semisupervised setting before calculating the accuracy
    '''
    optimizer.zero_grad()
    output = forward_call(data)
    if len(target.shape) == 1:
        target = torch_onehot(target, n_classes)
    # We use our own cross_entropy because it generalizes the F.nll_loss onto non-onehot labels
    error = cross_entropy(output, target)
    error.backward()
    optimizer.step()
    
    if not semisupervised:
        pred = output.data.max(1, keepdim=True)[1]
        target = output.data.max(1, keepdim=True)[1]
        correct = pred.eq(target).sum()
    else:
        # Semi-supervised training doesn't have a real notion of accuracy
        correct = -batch_size
    return error.item(), correct / batch_size
    
def test_classification(network, test_losses, test_accs, dataset, n_batches, forward_call, batch_size=8):
    '''
    Evaluate the model on the test dataset for some number of batches (or until the test dataset is exhausted).

    We test our model's performance using STANDARD classification. Recall that the augmentations only apply during training to
        'expand' the training dataset. Thus, regardless of whether we are pre-training or fine-tuning, our testing metrics
        are always 'how well does the model classify this dataset in question?'
    '''
    if forward_call is None:
        forward_call = network.forward
    data_loader = get_new_data_loader(dataset)
    test_loss = 0
    correct = 0
    network.eval()
    with torch.no_grad():
        for batch_counter, (data, target) in enumerate(data_loader):
            if batch_counter > n_batches or len(data) < batch_size:
                break
            output = forward_call(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

    seen_samples = batch_counter * batch_size
    test_losses.append(test_loss / batch_counter)
    test_accs.append(correct / seen_samples)


def train(
    network,
    forward_call,
    optimizer,
    semisupervised=False,
    n_batches=5000,
    n_classes=-1,
    augmentation=None,
    train_dataset=None,
    test_dataset=None,
    test_during_training=False,
    test_forward_call=None,
    plot_train_curves=False,
    batch_size=8
):
    '''
    Train the model (either pretraining or finetuning).
    '''
    losses, accs = [], []
    test_losses, test_accs, test_inds = [], [], []
    
    batch_idx = 0
    with tqdm(total=n_batches) as pbar:
        while batch_idx < n_batches:
            network.train()
            train_loader = get_new_data_loader(train_dataset)
            for data, target in train_loader:  
                if semisupervised:
                    assert callable(augmentation)
                    data, target = augment(augmentation, data, target, n_classes=n_classes)

                # Do training step
                loss, acc = train_step(
                    data,
                    target,
                    forward_call,
                    optimizer,
                    n_classes,
                    batch_size=batch_size,
                )
                losses.append(loss)
                accs.append(acc)

                if test_during_training and batch_idx > 0 and batch_idx % 500 == 0:
                    network.eval()
                    test_classification(network, test_losses, test_accs, test_dataset, int(n_batches/10), test_forward_call)
                    test_inds.append(len(losses))
                batch_idx += 1
                if batch_idx >= n_batches:
                    break
                pbar.update(1)
        
    test_classification(network, test_losses, test_accs, test_dataset, n_batches, test_forward_call)
    test_inds.append(len(losses))
    losses, accs = np.array(losses), np.array(accs)
    test_losses, test_accs = np.array(test_losses), np.array(test_accs)
    test_inds = np.array(test_inds)

    if plot_train_curves:
        plot_train_test(losses, test_losses, test_inds, titles=['Loss'])
    return losses, accs, test_losses, test_accs


