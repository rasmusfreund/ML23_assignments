import numpy as np


def one_in_k_encoding(vec, k):
    """One-in-k encoding of vector to k classes

    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc


def softmax(X):
    """
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array).

    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True
        option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate

    More precisely this is what you must do.

    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max
        i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that

    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation
            of the corresponding row in X i.e., res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    # Compute max value for each row
    max_vals = np.max(X, axis=1, keepdims=True)

    # Compute log of denominator
    log_denominator = np.log(np.sum(np.exp(X - max_vals), axis=1, keepdims=True))

    # Compute log of softmax and exponentiate
    log_softmax = X - max_vals - log_denominator
    res = np.exp(log_softmax)
    ### END CODE
    return res


def relu(x):
    """Compute the relu activation function on every element of the input

    Args:
        x: np.array
    Returns:
        res: np.array same shape as x
    Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    ### END CODE
    return np.maximum(0, x)


def make_dict(W1, b1, W2, b2):
    """Trivial helper function"""
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def get_init_params(input_dim, hidden_size, output_size):
    """Initializer function using Xavier/he et al Delving Deep into Rectifiers:
        Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(
        0, np.sqrt(2.0 / (input_dim + hidden_size)), size=(input_dim, hidden_size)
    )
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(
        0, np.sqrt(4.0 / (hidden_size + output_size)), size=(hidden_size, output_size)
    )
    b2 = np.zeros((1, output_size))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


class NetClassifier:
    def __init__(self):
        """Trivial Init"""
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """Compute class prediction for all data points in class X

        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None
        ### YOUR CODE HERE

        # Extract parameters
        W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

        # Forward pass
        Z1 = X.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = softmax(Z2)

        predictions = np.argmax(A2, axis=1)

        ### END CODE
        return predictions.reshape((1, -1))

    def score(self, X, y, params=None):
        """Compute accuracy of model on data X with labels y (mean 0-1 loss)

        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            acc: float, number of correct predictions divided by n.
                NOTE: This is accuracy, not in-sample error!
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        predictions = self.predict(X, params)
        acc = np.mean(predictions.flatten() == y.flatten())
        ### END CODE
        return acc

    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results
        and then implement the backwards pass using the intermediate stored results

        Use the derivative for cost as a function for input to softmax as derived above

        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation

        Returns
            cost: scalar - average cross entropy cost with weight decay parameter c
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]

        """
        cost = 0
        W1 = params["W1"]
        b1 = params["b1"]
        W2 = params["W2"]
        b2 = params["b2"]
        d_w1 = None
        d_w2 = None
        d_b1 = None
        d_b2 = None
        labels = one_in_k_encoding(y, W2.shape[1])  # shape n x k

        ### YOUR CODE HERE - FORWARD PASS - compute cost with weight decay and store relevant values for backprop
        # Number of samples
        n = X.shape[0]

        # Forward pass
        Z1 = X.dot(W1) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2) + b2
        A2 = softmax(Z2)

        # Compute cost
        cost = -np.sum(labels * np.log(A2)) / n

        # Add regularization
        cost += c * (np.sum(W1 ** 2) + np.sum(W2 ** 2)) / (2 * n)
        ### END CODE

        ### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all weights and bias, store them in d_w1, d_w2, d_b1, d_b2
        # Gradients of output
        d_Z2 = A2 - labels
        d_w2 = A1.T.dot(d_Z2) / n + c * W2 / n
        d_b2 = np.sum(d_Z2, axis=0, keepdims=True) / n

        d_A1 = d_Z2.dot(W2.T)
        d_Z1 = d_A1 * (Z1 > 0)  # RelU derivative
        d_w1 = X.T.dot(d_Z1) / n + c * W1 / n
        d_b1 = np.sum(d_Z1, axis=0, keepdims=True) / n

        ### END CODE
        # the return signature
        return cost, {"d_w1": d_w1, "d_w2": d_w2, "d_b1": d_b1, "d_b2": d_b2}

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        init_params,
        batch_size=32,
        lr=0.1,
        c=1e-4,
        epochs=30,
    ):
        """Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for
            Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working

        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter
           epochs: scalar - number of iterations through the data to use

        Sets:
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array
                of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        self.params = init_params
        W1 = init_params["W1"]
        b1 = init_params["b1"]
        W2 = init_params["W2"]
        b2 = init_params["b2"]
        hist = {
            "train_loss": np.zeros(epochs),
            "train_acc": np.zeros(epochs),
            "val_loss": np.zeros(epochs),
            "val_acc": np.zeros(epochs),
        }

        ### YOUR CODE HERE
        n_train = X_train.shape[0]

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.arange(n_train)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            # Mini-batch gradient descent
            for i in range(0, n_train, batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Compute cost and gradient
                cost, grads = self.cost_grad(X_batch, y_batch, self.params, c)

                # Update parameters
                self.params["W1"] -= lr * grads["d_w1"]
                self.params["b1"] -= lr * grads["d_b1"]
                self.params["W2"] -= lr * grads["d_w2"]
                self.params["b2"] -= lr * grads["d_b2"]

            # Compute loss and accuracy
            train_loss, _ = self.cost_grad(X_train, y_train, self.params, c)
            train_acc = self.score(X_train, y_train)
            val_loss, _ = self.cost_grad(X_val, y_val, self.params, c)
            val_acc = self.score(X_val, y_val)

            # Store performance
            hist["train_loss"][epoch] = train_loss
            hist["train_acc"][epoch] = train_acc
            hist["val_loss"][epoch] = val_loss
            hist["val_acc"][epoch] = val_acc

            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f},\nTrain Acc: {train_acc:.4f},\nVal Loss: {val_loss:.4f},\nVal Acc: {val_acc:.4f}"
            )
            ### END CODE
        # hist dict should look like this with something different than none
        # hist = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}
        ## self.params should look like this with something better than none, i.e. the best parameters found.
        # self.params = {'W1': None, 'b1': None, 'W2': None, 'b2': None}
        return hist


def numerical_grad_check(f, x, key):
    """Numerical Gradient Checker"""
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        dim = it.multi_index
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus - cminus) / (2 * h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert (
            np.abs(num_grad - grad[dim]) < eps
        ), "numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}".format(
            dim, num_grad, grad[dim]
        )
        it.iternext()


def test_grad():
    stars = "*" * 5
    print(stars, "Testing  Cost and Gradient Together")
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print("\n", stars, "Test Cost and Gradient of b2", stars)
    numerical_grad_check(f, params["b2"], "d_b2")
    print(stars, "Test Success", stars)

    print("\n", stars, "Test Cost and Gradient of w2", stars)
    numerical_grad_check(f, params["W2"], "d_w2")
    print("Test Success")

    print("\n", stars, "Test Cost and Gradient of b1", stars)
    numerical_grad_check(f, params["b1"], "d_b1")
    print("Test Success")

    print("\n", stars, "Test Cost and Gradient of w1", stars)
    numerical_grad_check(f, params["W1"], "d_w1")
    print("Test Success")


if __name__ == "__main__":
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)
    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, c=0)
    test_grad()
