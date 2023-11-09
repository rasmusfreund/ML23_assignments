import numpy as np
import matplotlib.pyplot as plt

from utils import running_mean, np_onehot
from augmentations import augment, AUGMENTATION_DICT
from get_data import PRETRAIN_SUBSAMPLE_SIZES

def plot_images(data_loader, batch_size, augmentation_name, n_classes):
    train_generator = enumerate(data_loader)
    _, (samples, targets) = next(train_generator)
    augmentation = AUGMENTATION_DICT[augmentation_name]
    aug_samples, interpolated_labels = augment(augmentation, samples, targets, n_classes)
    samples = np.concatenate([samples, aug_samples], axis=0)
    fig = plt.figure()
    r = 6
    for i in range(r):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(samples[i][0], cmap='gray', interpolation='none')
        if i < 3:
            plt.title("Ground Truth: {}".format(targets[i]))
        else:
            labels = interpolated_labels[i - 3]
            pos_labels = np.where(labels > 0)[0]
            if augmentation_name == 'no_aug':
                assert len(pos_labels) == 1
                plot_title = 'No Augmentation'
            else:
                assert 0 < len(pos_labels) <= 2
                if len(pos_labels) == 2:
                    plot_title = '{}:\nclass {} and {}'.format(augmentation_name, pos_labels[0], pos_labels[1])
                else:
                    plot_title = '{}:\nclass {} and {}'.format(augmentation_name, pos_labels[0], pos_labels[0])
            plt.title(plot_title)

        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()


def plot_train_test(
    all_train_vals,
    all_test_vals=None,
    all_test_inds=None,
    titles=None,
    color='blue',
    loc='upper right'
):
    """
    Plot train/test curves for a single training loop.
    """
    all_train_vals = np.array(all_train_vals)
    if all_test_vals is None and all_test_inds is None:
        all_test_vals = np.ones_like(all_train_vals) * -1
        all_test_inds = np.ones_like(all_train_vals) * -1
    else:
        all_test_vals, all_test_inds = np.array(all_test_vals), np.array(all_test_inds)
    if len(all_train_vals.shape) == 1:
        all_train_vals = np.expand_dims(all_train_vals, 0)
        assert len(all_test_vals.shape) == 1 and len(all_test_inds.shape) == 1
        all_test_vals = np.expand_dims(all_test_vals, 0)
        all_test_inds = np.expand_dims(all_test_inds, 0)
    if titles is not None:
        assert len(titles) == len(all_train_vals)
          
    fig = plt.figure()
    leg = []
    for i, (train_vals, test_vals, test_inds) in enumerate(zip(all_train_vals, all_test_vals, all_test_inds)):
        plt.subplot(1, len(all_train_vals), i+1)
        if np.all(train_vals >= 0):
            leg += ['Train Vals', 'Running Mean']
            plt.plot(np.arange(len(train_vals)), train_vals, color=color, alpha=0.3)
            plt.plot(np.arange(len(train_vals)), running_mean(train_vals), color=color)
        if np.all(test_vals >= 0):
            # No test set means that test losses should be all -1
            leg += ['Test Vals']
            plt.plot(test_inds, test_vals, color='red')
            plt.scatter(test_inds, test_vals, color='red')
        if i == 0:
            plt.legend(leg, loc=loc)
        plt.xlabel('Batches Seen')
        if titles is not None:
            plt.title(titles[i])
    plt.show()
    plt.close()

def plot_augmentation_results(all_aug_results, pretrain_data_str, finetune_data_str):
    """
    Given results across data samples for the three augmentations, plot results that show finetune accuracies.
    """
    plt.xscale('log')
    plt.xticks(PRETRAIN_SUBSAMPLE_SIZES, labels=PRETRAIN_SUBSAMPLE_SIZES)
    min_y_val = np.min([
        np.min(all_aug_results['no_aug']),
        np.min(all_aug_results['mixup']),
        np.min(all_aug_results['collage'])
    ])
    max_y_val = np.max([
        np.max(all_aug_results['no_aug']),
        np.max(all_aug_results['mixup']),
        np.max(all_aug_results['collage'])
    ])
    plt.ylim([min_y_val * 0.9, max_y_val * 1.1])

    plt.ylabel('Finetune accuracy on {} dataset'.format(finetune_data_str))
    plt.xlabel('Number of samples per class for {} pretraining'.format(pretrain_data_str))

    plt.plot(PRETRAIN_SUBSAMPLE_SIZES, all_aug_results['no_aug'], c='b', marker='.', label='No Aug.')
    plt.plot(PRETRAIN_SUBSAMPLE_SIZES, all_aug_results['mixup'], c='g', marker='.', label='Mixup')
    plt.plot(PRETRAIN_SUBSAMPLE_SIZES, all_aug_results['collage'], c='r', marker='.', label='Collage')
    plt.legend()
    plt.show()
