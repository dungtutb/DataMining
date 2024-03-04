import numpy as np


class DecisionTreeNode:
    def __init__(
        self, feature_index=None, threshold=None, left=None, right=None, value=None
    ):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes, this will be the predicted class


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        try:
            self.root = self._build_tree(X, y, depth=0)
        except ValueError as e:
            print("Error building decision tree:", e)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if depth == self.max_depth or n_classes == 1 or n_samples <= 1:
            return DecisionTreeNode(value=np.argmax(np.bincount(y.astype(int))))

        # Find the best split
        best_feature_index = None
        best_threshold = None
        best_entropy = float("inf")

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue  # Skip if one of the sides is empty
                left_entropy = self._calculate_entropy(y[left_indices])
                right_entropy = self._calculate_entropy(y[right_indices])
                total_entropy = (len(left_indices) / n_samples) * left_entropy + (
                    len(right_indices) / n_samples
                ) * right_entropy
                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    best_feature_index = feature_index
                    best_threshold = threshold

        if best_feature_index is None:
            raise ValueError("No suitable split found for the current node")

        # Split the data
        left_indices = np.where(X[:, best_feature_index] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature_index] > best_threshold)[0]

        # Recursively build left and right subtrees
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return DecisionTreeNode(
            feature_index=best_feature_index,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _predict_single(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def _calculate_entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy


class RandomForestClassifier:
    def __init__(self, base_classifier, n_estimators=10, max_depth=5):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = self.base_classifier(self.max_depth)
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            bootstrap_X = X[bootstrap_indices]
            bootstrap_y = y[bootstrap_indices]
            tree.fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.array(
            [
                np.argmax(np.bincount(tree_predictions))
                for tree_predictions in predictions.T
            ]
        )


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    # Create a synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=10,
        n_clusters_per_class=2,
        random_state=42,
    )

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize and train Random Forest
    random_forest_model = RandomForestClassifier(
        base_classifier=DecisionTree,
        n_estimators=20,
        max_depth=5,
    )
    random_forest_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = random_forest_model.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
