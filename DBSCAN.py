import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_points=5):
        self.eps = eps
        self.min_points = min_points

    def fit(self, X):
        self.X = X
        self.labels = [0] * len(X)
        self.cluster_id = 0

        for i in range(len(X)):
            if self.labels[i] != 0:
                continue
            neighbors = self._region_query(i)  # get neighborhoods of p
            if len(neighbors) < self.min_points:
                self.labels[i] = -1  # Noise point
            else:
                self.cluster_id += 1  # new core object p
                self._expand_cluster(i, neighbors)

    # find all neighbors of p
    def _region_query(self, index):
        neighbors = []
        for i in range(len(self.X)):
            if np.linalg.norm(self.X[index] - self.X[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    # find all density-reachable
    def _expand_cluster(self, index, neighbors):
        self.labels[index] = self.cluster_id
        i = 0
        while i < len(neighbors):  # stop when have not new neighbors added to cluster
            neighbor_index = neighbors[i]
            if self.labels[neighbor_index] == -1:
                self.labels[neighbor_index] = self.cluster_id
            elif self.labels[neighbor_index] == 0:
                self.labels[neighbor_index] = self.cluster_id
                new_neighbors = self._region_query(neighbor_index)
                if len(new_neighbors) >= self.min_points:
                    neighbors.extend(new_neighbors)  # merge to cluster of p
            i += 1


# Example usage
D = np.array([[1, 2], [2, 3], [8, 7], [8, 8], [7, 8], [2, 2], [7, 7], [3, 2]])
eps = 2
MinPts = 3
dbscan = DBSCAN(eps=eps, min_points=MinPts)
dbscan.fit(D)
print("Labels:", dbscan.labels)
