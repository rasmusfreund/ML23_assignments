# test on wine
# test on au digits
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from softmax import SoftmaxClassifier
from h1_util import export_fig, print_score, load_digits_train_data, load_digits_test_data
from argparse import ArgumentParser




def wine_test(epochs=200, batch_size=16, lr=0.1, normalize=True):
    print('wine test: params - epochs {0}, batch_size: {1}, learning rate: {2}'.format(epochs, batch_size, lr))
    features, target = load_wine(return_X_y=True)
    s = SoftmaxClassifier(num_classes=3)
    # Make a train/test split using 30% test size
    RANDOM_STATE = 42
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.25, random_state=RANDOM_STATE)
    if normalize:
        sc = StandardScaler() # makes every features zero mean standard deviation 1 
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test)
    X_train = np.c_[np.ones(X_train.shape[0]), X_train] # adds bias var
    print("X_train: ", X_train.shape)
    X_test = np.c_[np.ones(X_test.shape[0]), X_test] # adds bias vas

    s.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    print('Softmax Wine Classifier')
    print_score(s, X_train, X_test, y_train, y_test)
    hist = s.history
    fig, ax = plt.subplots()
    ax.plot(np.array(range(1, 1 + len(hist))), hist, 'b--', linewidth=2)
    ax.set_title('Cost as a function of epoch for wine data', fontsize=16)
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('Ein (1/n NLL)', fontsize=14)
    export_fig(fig, 'softmax_wine_cost_per_epoch.png')
    plt.show()
    return s

    
def digits_test(epochs=100, batch_size=32, lr=0.05):
    print('digits test: params - epochs {0}, batch_size: {1}, learning rate: {2}'.format(epochs, batch_size, lr))
    sc = SoftmaxClassifier(num_classes=10)
    X_train, y_train = load_digits_train_data()
    X_test, y_test = load_digits_test_data()
    sc.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)
    print_score(sc, X_train, X_test, y_train, y_test)
    fig, ax = plt.subplots()
    hist = sc.history
    ax.plot(np.array(range(1, 1 + len(hist))), hist, 'b-x')
    ax.set_xlabel('epoch', fontsize=14)
    ax.set_ylabel('Ein (1/n NLL)', fontsize=14)
    ax.set_title('softmax cost per epoch', fontsize=16)
    export_fig(fig, 'softmax_cost_per_epoch.png')
    plt.show()

def show_digits(grid=(4, 4)):
    X_train, y_train = load_digits_train_data()
    fig, axes = plt.subplots(grid[0], grid[1])
    X_train, y_train = load_digits_train_data()
    n = X_train.shape[0]
    ridx = np.random.randint(0, n, size=grid)
    for i in range(grid[0]):
        for j in range(grid[1]):
            axes[i][j].imshow(X_train[ridx[i][j],].reshape(28, 28), cmap=plt.get_cmap('bone'))
            axes[i][j].set_title('{0}'.format(y_train[ridx[i][j]]))
            axes[i][j].axis('off')
    plt.show()        
    
    
def digits_visualize(epochs=1, batch_size=64, lr=0.01):
    sc = SoftmaxClassifier(num_classes=10)
    X_train, y_train = load_digits_train_data()
    sc.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, lr=lr)    
    w = sc.W
    rs = w.reshape(28, 28, 10, order='F')
    rs2 = np.transpose(rs, axes=[1,0,2])
    fig, ax = plt.subplots()
    ax.imshow(rs2.reshape(28, -1, order='F'), cmap='bone')
    ax.set_title('digits weight vector visualized')
    export_fig(fig, 'softmax_weight_vector.png')
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-wine', action='store_true', default=False)
    parser.add_argument('-digits', action='store_true', default=False)
    parser.add_argument('-visualize', action='store_true', default=False)
    parser.add_argument('-show_digits', action='store_true', default=False)    
    parser.add_argument('-lr', dest='lr', type=float, default=-1)
    parser.add_argument('-bs', type=int, dest='batch_size', default=-1)
    parser.add_argument('-epochs', dest='epochs', type=int, default=-1)    
    args = parser.parse_args()
    print('vars args', vars(args))
    kwargs = {}
    if args.lr >= 0:
        kwargs['lr'] = args.lr
    if args.batch_size >=0:
        kwargs['batch_size'] = args.batch_size
    if args.epochs >=0:
        kwargs['epochs'] = args.epochs            
    if args.wine:
        wine_test(**kwargs)
    if args.digits:
        digits_test(**kwargs)
    if args.visualize:
        digits_visualize(**kwargs)
    if args.show_digits:
        show_digits()
