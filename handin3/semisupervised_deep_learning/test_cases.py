from augmentations import test_augmentations
from network import test_network
from utils import test_cross_entropy

if __name__ == '__main__':
    test_augmentations()

    print('Due to randomness, the network tests may occassionally fail on correct implementations. If so, try to rerun a few times to see if the problem persists.')
    data_str = 'mnist'
    test_network(n_samples=1, n_batches=500, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)
    test_network(n_samples=4, n_batches=1000, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)
    test_network(n_samples=16, n_batches=2000, minimum_acc=0.9, maximum_loss=0.05, data_str=data_str)

    data_str = 'emnist'
    test_network(n_samples=1, n_batches=1000, minimum_acc=0.9, maximum_loss=0.5, data_str=data_str)
    test_network(n_samples=4, n_batches=2000, minimum_acc=0.9, maximum_loss=0.5, data_str=data_str)
    test_network(n_samples=16, n_batches=4000, minimum_acc=0.9, maximum_loss=1.0, data_str=data_str)

    test_cross_entropy()

    print(
        'All tests passed! '
        'Warning -- the testing infrastructure is not comprehensive. '
        'You may pass the test cases but still train poorly due to an issue that was not checked for in the tests.'
    )
