from utils import np_onehot
import numpy as np
import torch

def collage(batch_i, batch_j):
    """
    Take top-left and bottom-right quarters from batch_j and top-right and bottom-left quarters from batch_i

    Since this takes one half from one image and another half from another, our class interpolation should be 0.5

    The inputs are numpy, not torch
    """
    # Needs to be a square image to apply collage augmentation
    im_size = int(batch_i.shape[1])
    assert im_size == int(batch_i.shape[2])
    result = None
    interpolation = None
    
    ### YOUR CODE HERE
    ### END CODE

    return result, interpolation

def mixup(batch_i, batch_j, alpha=0.3):
    """
    Linearly interpolate between two images. Intention is (interpolation * batch_i + (1-interpolation)*batch_j)
    We find the interpolation value in [0, 1] for you by sampling from a Beta distribution; see https://arxiv.org/pdf/1710.09412.pdf

    The inputs are numpy, not torch
    """
    interpolation = np.random.beta(alpha, alpha)
    result = None

    ### YOUR CODE HERE
    ### END CODE

    return result, interpolation

def no_aug(batch_i, batch_j):
    return batch_i, 1

AUGMENTATION_DICT = {
    'mixup': mixup,
    'collage': collage,
    'no_aug': no_aug
}

def augment(augmentation, batch, labels, n_classes):
    """
    Apply the augmentation to the batch in question.

    The mixup and collage augmentations will combine two images to make something 'in between' them.
    The label should then be a linear combination of the two image labels.
    """
    if augmentation is None:
        augmentation = no_aug

    batch_size = len(batch)
    new_batch = np.zeros_like(batch)
    merge_indices = np.zeros(batch_size, dtype=np.int32)
    interpolations = np.zeros(batch_size)
    for i, image in enumerate(batch):
        # merge_indices[i] is the index of the element that batch[i] will be augmented with
        # We do not allow merge_indices[i] = i
        merge_indices[i] = np.random.choice(np.delete(np.arange(batch_size), i))        

        # Apply the augmentation
        new_batch[i], interpolations[i] = augmentation(batch[i], batch[merge_indices[i]])
        
    images = torch.tensor(new_batch)
    interpolations = np.expand_dims(interpolations, 1) # for correct broadcasting
    merged_labels = labels[merge_indices]

    targets = interpolations * np_onehot(labels, n_classes)
    targets += (1 - interpolations) * np_onehot(merged_labels, n_classes)
    targets = torch.tensor(targets)

    return images, targets


def test_augmentations():
    im_1 = np.ones((1, 4, 4), dtype=np.float32)
    im_2 = np.zeros((1, 4, 4), dtype=np.float32)

    collage_output, collage_interpolation = collage(im_1, im_2)
    mixup_output, mixup_interpolation = mixup(im_1, im_2)

    np.testing.assert_array_equal(collage_output[0, :2, :2], 0)
    np.testing.assert_array_equal(collage_output[0, 2:, :2], 1)
    np.testing.assert_array_equal(collage_output[0, :2, 2:], 1)
    np.testing.assert_array_equal(collage_output[0, 2:, 2:], 0)
    np.testing.assert_equal(collage_interpolation, 0.5)

    np.testing.assert_array_equal(mixup_output, np.ones_like(mixup_output) * mixup_interpolation)
    np.testing.assert_array_less(mixup_interpolation, 1)
    np.testing.assert_array_less(0, mixup_interpolation)


if __name__ == '__main__':
    test_augmentations()
