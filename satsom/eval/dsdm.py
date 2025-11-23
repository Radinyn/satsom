import torch
import torch.nn as nn
import torch.nn.functional as F


class DSDMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        n_classes,
        T=2.0,
        ema=0.01,
        device="cuda" if torch.cuda.is_available() else "cpu",
        p_norm="fro",
        pruning=False,
        max_addresses=None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.n_classes = n_classes
        self.T = T
        self.ema = ema
        self.p_norm = p_norm
        self.pruning = pruning
        self.max_addresses = max_addresses  # integer or None

        # initialize memory with one dummy address
        self.Address = torch.zeros(1, input_dim, device=self.device)
        self.M = torch.zeros(1, n_classes, device=self.device)

        # global running error for dynamic expansion
        self.global_error = torch.tensor(0.0, device=self.device)

    def _compute_distances(self, X):
        """
        X: (B, D)
        Address: (N, D)
        returns: (B, N) distances
        """
        A = self.Address
        return torch.sqrt(
            -2 * torch.mm(X, A.t())
            + torch.sum(A**2, dim=1).unsqueeze(0)
            + torch.sum(X**2, dim=1).unsqueeze(1)
        )

    # ---------------------------------------------------------
    # Predict
    # ---------------------------------------------------------
    def forward(self, x):
        """
        x: (B, D) or (B, C, H, W)
        returns: logits (B, n_classes)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        with torch.no_grad():
            dist = self._compute_distances(x)  # (B, N)
            soft_w = F.softmin(dist / self.T, dim=1)  # (B, N)
            logits = torch.matmul(soft_w, self.M)  # (B, C)
        return logits

    # shorthand
    def predict(self, x):
        return self.forward(x)

    def partial_fit(self, x, y):
        """
        x: (B, D)
        y: (B) integer labels
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        x = x.to(self.device)
        y = y.to(self.device)

        B = x.size(0)
        for i in range(B):
            xi = x[i]
            yi = y[i]

            # distance to all addresses
            dist = torch.norm(xi - self.Address, p=2, dim=1)
            min_val, min_idx = torch.min(dist, dim=0)

            # smooth global error update
            self.global_error = (1 - self.ema) * self.global_error + self.ema * min_val

            # create a new address if too far
            if min_val >= self.global_error:
                yi_onehot = (
                    F.one_hot(yi, num_classes=self.n_classes).float().to(self.device)
                )
                self.Address = torch.cat([self.Address, xi.unsqueeze(0)], dim=0)
                self.M = torch.cat([self.M, yi_onehot.unsqueeze(0)], dim=0)

                # optional pruning
                if self.pruning and self.max_addresses is not None:
                    if len(self.Address) > self.max_addresses:
                        self._prune_simple()

            else:
                # SOM-like update
                delta = xi - self.Address  # (N, D)
                w = F.softmin(dist / self.T, dim=0)  # (N)
                self.Address += self.ema * (w.unsqueeze(1) * delta)
                target_onehot = F.one_hot(yi, self.n_classes).float().to(self.device)
                self.M += self.ema * (w.unsqueeze(1) * (target_onehot - self.M))

    def _prune_simple(self):
        """
        Keeps only the most recently updated addresses.
        (More advanced LOF or class-balanced pruning is possible.)
        """
        keep = self.max_addresses
        self.Address = self.Address[-keep:]
        self.M = self.M[-keep:]
