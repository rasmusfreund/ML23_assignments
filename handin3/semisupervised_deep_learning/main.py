import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
from copy import deepcopy
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
from augmentations import collage, mixup, no_aug, augment, AUGMENTATION_DICT
from plotting import plot_images, plot_augmentation_results
from network_training import train
from network import Net

import argparse

if __name__ == '__main__':
    # Command-line arguments that let you control training and visualization parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--finetune-dataset',
        required=True, # This flag means that this is a required parameter
        type=str, # This flag describes the expected type of the parameter
        help='Dataset to fine-tune on'
    )
    parser.add_argument(
        '--pre-train-dataset',
        type=str,
        default='mnist',
        help='Dataset to run pre-training on'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size to run training with'
    )
    parser.add_argument(
        '--test-during-training',
        action='store_true', # If `--test-during-training` is included in the command, then this is set to True.
        #                      Otherwise, it is set to false.
        help='If present, will evaluate classification performance on the test set every 500 training batches. '
        'This is useful if you also have --plot-train-curves present, as you will then have train/test curves to look at as training progresses.'
    )
    parser.add_argument(
        '--plot-augmentations',
        action='store_true',
        help='If present, visualize augmented samples from the training dataset'
    )
    parser.add_argument(
        '--plot-train-curves',
        action='store_true',
        help='If present, plot the curve at the end of each training loop'
    )
    parser.add_argument(
        '--n-batches-pre-train',
        type=int,
        default=5000,
        help='Number of batches to run when pre-training'
    )
    parser.add_argument(
        '--n-batches-finetune',
        type=int,
        default=1000,
        help='Number of batches to do run when finetuning'
    )
    # Now put all of the arguments into a namespace called `args`
    args = parser.parse_args()

    # Get the datasets that we will pre-train and fine-tune on
    pre_train_subsampled_datasets, pre_train_test_data  = get_dataset(args.pre_train_dataset)
    finetune_subsampled_datasets, finetune_test_data  = get_dataset(args.finetune_dataset)

    # Test to make sure that the subsampled datasets actually have the desired number of samples per class
    test_data_balance(pre_train_subsampled_datasets, args.pre_train_dataset)
    test_data_balance(finetune_subsampled_datasets, args.finetune_dataset)

    # Convenience function if you would like to look at the augmentations
    # Make sure to include `--plot-augmentations` when running this file if you'd like to see the augmentations
    if args.plot_augmentations:
        data_vis_loader = get_new_data_loader(pre_train_test_data, batch_size=3)
        n_classes = NUM_CLASSES_DICT[args.pre_train_dataset]
        plot_images(data_vis_loader, batch_size=args.batch_size, augmentation_name='no_aug', n_classes=n_classes)
        plot_images(data_vis_loader, args.batch_size, augmentation_name='collage', n_classes=n_classes)
        plot_images(data_vis_loader, args.batch_size, augmentation_name='mixup', n_classes=n_classes)

    all_augmentation_results = {}
    for aug_str, augmentation in AUGMENTATION_DICT.items():
        # If we have an augmentation, then our class labels are no longer onehot.
        # As a result, we cannot do things like evaluate prediction accuracy easily.
        # To account for this, we create the `semisupervised` flag. When True, the training/testing functions will know
        #   that we do not have onehot labels.
        print('\n')
        print('Now using augmentation {}\n'.format(aug_str))
        if aug_str != 'no_aug':
            semisupervised = True
        else:
            semisupervised = False

        augmentation_results = []

        # For each dataset size, run pre-training and fine-tuning to evaluate performance on this augmentation
        for pre_train_samples_per_class in PRETRAIN_SUBSAMPLE_SIZES:
            print('Pretraining on {} with samples_per_class={}'.format(args.pre_train_dataset, pre_train_samples_per_class))

            # Get the pre-training dataset with the appropriate number of samples per class
            pre_train_data_subsample = pre_train_subsampled_datasets[pre_train_samples_per_class]

            # Instantiate our network
            network = Net(
                pre_train_classes=NUM_CLASSES_DICT[args.pre_train_dataset],
                generalization_classes=NUM_CLASSES_DICT[args.finetune_dataset]
            )

            # Torch optimizer that will allow our network to learn
            pre_train_optimizer = optim.SGD(network.parameters(), lr=0.01)

            # Run pre-training and store the losses/accuracies over training in variables
            pretrain_train_losses, pretrain_train_accs, pretrain_test_losses, pretrain_test_accs = train(
                network,
                network.forward,
                pre_train_optimizer,
                semisupervised=semisupervised,
                augmentation=augmentation,
                train_dataset=pre_train_data_subsample,
                test_dataset=pre_train_test_data,
                test_during_training=args.test_during_training,
                n_batches=args.n_batches_pre_train,
                n_classes=NUM_CLASSES_DICT[args.pre_train_dataset],
                plot_train_curves=args.plot_train_curves
            )

            # Get the finetuning dataset and optimizer
            data_subsample = finetune_subsampled_datasets[FINETUNE_SUBSAMPLE_SIZE]
            optimizer = optim.SGD(network.generalizer.parameters(), lr=0.01)

            # Run finetuning and store the losses/accuracies over training in variables
            print('Finetuning on {}'.format(args.finetune_dataset))
            finetune_train_losses, finetune_train_accs, finetune_test_losses, finetune_test_accs = train(
                network,
                network.generalize,
                optimizer,
                augmentation=None,
                train_dataset=data_subsample,
                test_dataset=finetune_test_data,
                test_during_training=args.test_during_training,
                test_forward_call=network.generalize,
                n_batches=args.n_batches_finetune,
                n_classes=NUM_CLASSES_DICT[args.finetune_dataset],
                plot_train_curves=args.plot_train_curves
            )
            augmentation_results.append(finetune_test_accs[-1])

        augmentation_results = np.array(augmentation_results)
        all_augmentation_results[aug_str] = augmentation_results

    plot_augmentation_results(all_augmentation_results, args.pre_train_dataset, args.finetune_dataset)
