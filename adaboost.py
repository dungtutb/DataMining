import numpy as np
from sklearn.metrics import accuracy_score

np.set_printoptions(precision=3)


def entropy(ar):
    classes, counts = np.unique(ar, return_counts=True)
    probabilities = counts / len(ar)
    e = -np.sum(
        probabilities * np.log2(probabilities + 1e-10)
    )  # Add a small epsilon to avoid log(0)
    return e


class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        best_entropy = float("inf")
        n_samples = len(y)

        for i in range(n_samples):
            if i == 0:
                X_pre = 0
            else:
                X_pre = X[i - 1]
            if X[i] != X_pre:  # Skip duplicates
                threshold = round((X[i] + X_pre) / 2, 2)  # Midpoint threshold
                left_indices = np.where(X <= threshold)[0]
                right_indices = np.where(X > threshold)[0]
                left_labels = y[left_indices]
                right_labels = y[right_indices]
                left_entropy = entropy(y[left_indices])
                right_entropy = entropy(y[right_indices])
                total_entropy = (len(left_indices) / n_samples) * left_entropy + (
                    len(right_indices) / n_samples
                ) * right_entropy

                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    self.threshold = threshold
                    self.left_class = self._get_majority_class(left_labels)
                    self.right_class = self._get_majority_class(right_labels)
                    if self.left_class is None:
                        self.left_class = self.right_class
                    if self.right_class is None:
                        self.right_class = self.left_class

    def predict(self, X):
        return np.where(X > self.threshold, self.right_class, self.left_class)

    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-10)
        )  # Thêm epsilon nhỏ để tránh log(0)
        return entropy

    def _get_majority_class(self, ar):
        if len(ar) == 0:
            return None
        unique, counts = np.unique(ar, return_counts=True)
        return unique[np.argmax(counts)]


def bootstrap_sample(X, y, n_samples, replace=True):
    # Create training set Di by sampling from D
    indices = np.random.choice(X.shape[0], size=n_samples, replace=replace)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    sorted_indices = np.argsort(X_bootstrap)
    X_bootstrap = X_bootstrap[sorted_indices]
    y_bootstrap = y_bootstrap[sorted_indices]

    return X_bootstrap, y_bootstrap


class AdaBoost:
    def __init__(self, base_classifier, n_estimators=50):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators  # equivalent to k
        self.alphas = []
        self.classifiers = []

    def fit(self, X, y, n_samples, debug=False):

        # Initialize the weights for all N examples
        weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            if debug:
                print("\n===========> Round {} : <===========".format(i + 1))
            X_bootstrap, y_bootstrap = bootstrap_sample(X, y, n_samples, replace=True)
            if debug:
                print("X: {}".format(X_bootstrap.reshape(1, -1)))

            error = 1
            while error > 0.5:
                # Create training set Di by sampling from D

                classifier = self.base_classifier()
                classifier.fit(X_bootstrap, y_bootstrap)

                predictions = classifier.predict(X)
                # Caculate the weighted error
                error = np.sum(weights * (predictions != y)) / n_samples

            if debug:
                print("Threshold: {}".format(classifier.threshold))
                print("Y left: {}".format(classifier.left_class))
                print("Y right: {}".format(classifier.right_class))
                # print("Error: {}".format(error))
            # caculate alpha
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            if debug:
                print("Alpha: {}".format(alpha))

            # Update weights
            weights *= np.exp(
                -alpha * y_bootstrap * predictions
            )  # y_bootstrap and prediction is array of 1 or -1
            weights /= np.sum(weights)

            self.classifiers.append(classifier)

            if debug:
                print("Weights: {}".format(weights))
                print("Y: {}".format(predictions))

    def predict(self, X, debug=False):
        predictions = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * classifier.predict(X)
        if debug:
            print("Sum: {}".format(predictions))
        return np.sign(predictions)


# Example usage:
if __name__ == "__main__":
    from sklearn.metrics import accuracy_score

    # Create a dataset
    X = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # X = X.reshape(-1, 1)
    y = np.array([1, 1, 1, -1, -1, -1, -1, 1, 1, 1])

    print("X: {}".format(X.reshape(1, -1)))
    print("Y: {}".format(y))

    # # Split the dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    adaboost = AdaBoost(base_classifier=DecisionStumpClassifier, n_estimators=10)
    adaboost.fit(X, y, n_samples=X.shape[0], debug=True)  # Training

    print("\n++++++++++ Result ++++++++++")
    print("Y True: {}".format(y))
    y_pred = adaboost.predict(X, debug=True)
    print("Y Prediction: {}".format(y_pred))
    # Evaluate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")


""" Output
X: [[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]]
Y: [ 1  1  1 -1 -1 -1 -1  1  1  1]

===========> Round 1 : <===========
X: [[0.1 0.1 0.5 0.5 0.6 0.7 0.7 0.9 0.9 0.9]]
Threshold: 0.8
Y left: -1
Y right: 1
Alpha: 1.5890269139239728
Weights: [0.304 0.304 0.013 0.013 0.013 0.013 0.013 0.304 0.013 0.013]
Y: [-1 -1 -1 -1 -1 -1 -1 -1  1  1]

===========> Round 2 : <===========
X: [[0.1 0.2 0.3 0.4 0.6 0.8 0.9 0.9 1.  1. ]]
Threshold: 0.7
Y left: 1
Y right: 1
Alpha: 2.6403312046346197
Weights: [0.051 0.051 0.002 0.418 0.418 0.002 0.002 0.051 0.002 0.002]
Y: [1 1 1 1 1 1 1 1 1 1]

===========> Round 3 : <===========
X: [[0.2 0.2 0.5 0.5 0.5 0.6 0.7 0.8 0.9 0.9]]
Threshold: 0.75
Y left: -1
Y right: 1
Alpha: 2.276508602492993
Weights: [4.576e-01 4.576e-01 2.009e-04 3.947e-02 3.947e-02 2.009e-04 2.009e-04
 4.821e-03 2.009e-04 2.009e-04]
Y: [-1 -1 -1 -1 -1 -1 -1  1  1  1]

===========> Round 4 : <===========
X: [[0.1 0.1 0.2 0.2 0.5 0.6 0.6 0.8 0.8 0.9]]
Threshold: 0.35
Y left: 1
Y right: -1
Alpha: 3.778382896669932
Weights: [5.291e-03 5.291e-03 2.323e-06 8.734e-01 4.564e-04 2.323e-06 2.323e-06
 1.067e-01 4.445e-03 4.445e-03]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 5 : <===========
X: [[0.1 0.2 0.2 0.4 0.5 0.7 0.7 0.8 0.9 1. ]]
Threshold: 0.3
Y left: 1
Y right: -1
Alpha: 2.22444952154415
Weights: [4.913e-04 4.913e-04 2.157e-07 8.110e-02 4.238e-05 2.157e-07 2.157e-07
 8.473e-01 3.530e-02 3.530e-02]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 6 : <===========
X: [[0.1 0.1 0.1 0.5 0.6 0.7 0.9 1.  1.  1. ]]
Threshold: 0.8
Y left: -1
Y right: 1
Alpha: 1.18926351239149
Weights: [5.698e-04 5.698e-04 2.501e-07 8.717e-03 4.555e-06 2.318e-08 2.501e-07
 9.825e-01 3.795e-03 3.795e-03]
Y: [-1 -1 -1 -1 -1 -1 -1 -1  1  1]

===========> Round 7 : <===========
X: [[0.3 0.3 0.4 0.4 0.4 0.6 0.7 0.9 1.  1. ]]
Threshold: 0.8
Y left: -1
Y right: 1
Alpha: 1.107740634493969
Weights: [5.782e-04 5.782e-04 2.769e-08 9.651e-04 5.043e-07 2.567e-09 2.769e-08
 9.970e-01 4.201e-04 4.201e-04]
Y: [-1 -1 -1 -1 -1 -1 -1 -1  1  1]

===========> Round 8 : <===========
X: [[0.1 0.1 0.3 0.5 0.6 0.8 0.9 0.9 1.  1. ]]
Threshold: 0.7
Y left: 1
Y right: 1
Alpha: 4.622599559017819
Weights: [5.257e-05 5.257e-05 2.518e-09 9.087e-01 4.748e-04 2.334e-10 2.518e-09
 9.066e-02 3.820e-05 3.820e-05]
Y: [1 1 1 1 1 1 1 1 1 1]

===========> Round 9 : <===========
X: [[0.1 0.2 0.2 0.4 0.6 0.6 0.6 0.6 0.8 0.9]]
Threshold: 0.3
Y left: 1
Y right: -1
Alpha: 2.3466423753821766
Weights: [5.214e-05 5.214e-05 2.497e-09 9.012e-01 4.710e-04 2.315e-10 2.497e-09
 8.992e-02 4.137e-03 4.137e-03]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 10 : <===========
X: [[0.1 0.1 0.3 0.3 0.3 0.4 0.5 0.6 0.6 0.9]]
Threshold: 0.35
Y left: 1
Y right: -1
Alpha: 2.306784107135079
Weights: [5.702e-07 5.702e-07 2.731e-11 9.939e-01 5.194e-04 2.531e-12 2.731e-11
 9.833e-04 4.525e-05 4.563e-03]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

++++++++++ Result ++++++++++
Y True: [ 1  1  1 -1 -1 -1 -1  1  1  1]
Sum: [11.757 11.757 11.757 -9.556 -9.556 -9.556 -9.556 -5.003  2.769  2.769]
Y Prediction: [ 1.  1.  1. -1. -1. -1. -1. -1.  1.  1.]
Accuracy: 0.9

"""
