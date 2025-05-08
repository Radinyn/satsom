"""
A simple, efficient k-Nearest Neighbors classifier in PyTorch with incremental learning support.
Used to evaluate SatSOM performance.
"""

import torch


class KNNClassifier:
    """
    A simple, efficient k-Nearest Neighbors classifier in PyTorch with incremental learning support.

    Parameters
    ----------
    k : int
        Number of neighbors to use.
    device : str or torch.device, optional
        Device to store data and perform computation on. Defaults to CUDA if available.
    """

    def __init__(self, k=5, device=None):
        self.k = k
        self.device = (
            torch.device(device)
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.X_train = None
        self.y_train = None

    def partial_fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Incrementally add new data to the training set.

        Parameters
        ----------
        X : torch.Tensor, shape (n_samples, n_features)
            Feature matrix of new training samples.
        y : torch.Tensor, shape (n_samples,)
            Labels of new training samples.
        """
        X = X.to(self.device)
        y = y.to(self.device)
        if self.X_train is None:
            self.X_train = X
            self.y_train = y
        else:
            # concatenate along sample dimension
            self.X_train = torch.cat((self.X_train, X), dim=0)
            self.y_train = torch.cat((self.y_train, y), dim=0)

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        Fit the classifier by replacing any existing training data.
        Equivalent to resetting and performing a single partial_fit call.
        """
        self.X_train = None
        self.y_train = None
        self.partial_fit(X, y)

    def predict(self, X: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
        """
        Predict labels for query samples.

        Parameters
        ----------
        X : torch.Tensor, shape (n_queries, n_features)
            Query feature matrix.
        batch_size : int
            Batch size for distance computation to save memory.

        Returns
        -------
        torch.Tensor, shape (n_queries,)
            Predicted labels.
        """
        X = X.to(self.device)
        num_test = X.size(0)
        preds = []

        # process in batches
        for start in range(0, num_test, batch_size):
            end = min(start + batch_size, num_test)
            X_batch = X[start:end]
            # Efficient L2 distance: (a-b)^2 = a^2 + b^2 - 2ab
            x_norm = (X_batch**2).sum(dim=1).unsqueeze(1)  # [batch,1]
            train_norm = (self.X_train**2).sum(dim=1).unsqueeze(0)  # [1,n_train]
            dists = (
                x_norm + train_norm - 2.0 * X_batch @ self.X_train.t()
            )  # [batch, n_train]

            # retrieve k smallest distances
            _, idx = torch.topk(dists, self.k, dim=1, largest=False)
            knn_labels = self.y_train[idx]  # [batch, k]
            # majority vote (mode)
            mode_labels, _ = torch.mode(knn_labels, dim=1)
            preds.append(mode_labels)

        return torch.cat(preds)
