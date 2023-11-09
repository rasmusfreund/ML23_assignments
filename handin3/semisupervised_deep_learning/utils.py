import numpy as np
import torch

def torch_onehot(labels, n_classes):
    onehot_labels = torch.zeros((len(labels), n_classes))
    onehot_labels[torch.arange(len(labels)), labels] = 1
    return onehot_labels

def np_onehot(labels, n_classes):
    onehot_labels = np.zeros((len(labels), n_classes))
    onehot_labels[np.arange(len(labels)), labels] = 1
    return onehot_labels

def running_mean(vals, steps=30):
    means = np.zeros_like(vals)
    for i, loss in enumerate(vals):
        if i < steps:
            means[i] = np.mean(vals[:i])
        else:
            means[i] = np.mean(vals[i-steps:i])
    return means

def cross_entropy(pred, target):
    """
    This is a by-hand implementation of the cross-entropy loss.

    The reason we need this done by hand is that Pytorch's implementation assumes onehot targets.
    However, if we use an augmentation then we may have a target that is a linear combination of two classes.

    NOTE: pred and target both have shape [batch_size, n_classes]
    NOTE: we assume that the network's output layer has a LOG-SOFTMAX.

    Recall from lectures that the cross entropy loss for predictions p and label vector y is -SUM_i y_i ln(p_i). 
    For you, the vector p=pred already contains the values ln(p_i).
    """
    if not torch.is_tensor(pred) or not torch.is_tensor(target):
        raise ValueError('X-Entropy loss requires torch tensors for input')

    ### YOUR CODE HERE
    ### END CODE

    return mean_log_likelihoods


def test_cross_entropy():
    predictions = np.array([
        [[1, 0]],
        [[0.5, 0.5]],
        [[0, 1]],
        [[0.5, 0.5]]
    ], dtype=np.float32)
    targets = np.array([
        [[1, 0]],
        [[1, 0]],
        [[0.5, 0.5]],
        [[0.5, 0.5]]
    ], dtype=np.float32)
    correct_outputs = np.array([-1, -0.5, -0.5, -0.5])

    # Evaluate correctness of the loss on SINGLE pred-target inputs
    for pred, target, correct_output in zip(predictions, targets, correct_outputs):
        pred = torch.tensor(pred)
        target = torch.tensor(target)
        loss = cross_entropy(pred, target).numpy()
        np.testing.assert_equal(loss, correct_output)

    # Evaluate correctness of the loss on BATCHWISE pred-target inputs
    pred_batch = torch.squeeze(torch.from_numpy(predictions), dim=1)
    target_batch = torch.squeeze(torch.from_numpy(targets), dim=1)
    loss = cross_entropy(pred_batch, target_batch).numpy()
    np.testing.assert_equal(loss, np.mean(correct_outputs))

    loss = cross_entropy(pred_batch, target_batch).numpy()
    np.testing.assert_equal(loss, np.mean(correct_outputs))


if __name__ == '__main__':
    test_cross_entropy()
