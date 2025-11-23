# propre.py
import torch
from typing import Optional


class PROPREClassifier(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        nH: int = 30,
        n_classes: int = 10,
        sigma: float = None,
        kappa: float = 1.0,
        theta: float = 0.7,
        p: float = 10.0,
        lr_som: float = 0.1,
        lr_lr: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or device
        self.input_dim = input_dim
        self.nH_side = nH
        self.n_units = nH * nH
        self.n_classes = n_classes

        # SOM prototypes: (n_units, input_dim)
        # small init noise; in practice you may initialize from data
        self.wSOM = torch.randn(self.n_units, input_dim, device=self.device) * 0.01

        # Linear readout weights (n_classes, n_units)
        # Each row maps hidden activations to one class output (population-coded)
        self.wLR = torch.zeros(n_classes, self.n_units, device=self.device)

        # PROPRE parameters (paper-inspired defaults)
        self.kappa = kappa  # gaussian width for converting distances -> similarities
        self.theta = theta  # sparsity / activation threshold
        self.p = p  # exponent for TF
        self.lr_som = lr_som
        self.lr_lr = lr_lr

        # neighbourhood spatial scale for prototype updates (grid units)
        self.sigma = sigma if sigma is not None else max(1.0, nH / 10.0)

        # precompute coordinates of SOM units in 2D grid for neighborhood calculation
        coords = []
        for i in range(self.n_units):
            r = i // self.nH_side
            c = i % self.nH_side
            coords.append((r, c))
        self.coords = torch.tensor(
            coords, dtype=torch.float32, device=self.device
        )  # (n_units, 2)

    def _distances(self, x: torch.Tensor):
        """
        Compute euclidean distances between x and all prototypes.
        x: (B, D) or (D,)
        returns: (B, n_units)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # efficient pairwise squared distances
        XX = (x * x).sum(dim=1, keepdim=True)  # (B,1)
        WW = (self.wSOM * self.wSOM).sum(dim=1).unsqueeze(0)  # (1, n_units)
        D2 = XX + WW - 2.0 * (x @ self.wSOM.t())
        D2 = torch.clamp(D2, min=0.0)
        return torch.sqrt(D2)  # (B, n_units)

    def _gaussian(self, dists: torch.Tensor, kappa: float):
        """
        Convert distances to similarities via Gaussian: exp(-d^2 / (2*kappa^2))
        dists: (B, n_units)
        """
        return torch.exp(-(dists**2) / (2.0 * (kappa**2)))

    def _transfer_function(self, g_tilde: torch.Tensor):
        """
        The paper uses TF_{theta,p} which sparsifies activations.
        Implementation:
            - compute per-sample max across units
            - Tilde^p / max^{p-1}, then threshold by theta
        g_tilde: (B, n_units) values in (0,1]
        returns: zH (B, n_units)
        """
        g_max = g_tilde.max(dim=1, keepdim=True)[0] + 1e-12  # (B,1)
        # exponentiate and normalize
        g_p = (g_tilde**self.p) / (g_max ** (self.p - 1))
        # threshold by theta (keep values only if > theta relative to max)
        mask = (
            g_p / (g_max ** (self.p - 1) + 1e-12)
        ) > self.theta  # equivalent condition
        # safer: threshold original ratio g_tilde/g_max > theta
        mask = (g_tilde / g_max) > self.theta
        zH = g_p * mask.float()
        return zH

    def predict(self, x: torch.Tensor):
        """
        Returns logits for x (B, n_classes)
        """
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        d = self._distances(x)  # (B, n_units)
        g_tilde = self._gaussian(d, self.kappa)  # (B, n_units)
        zH = self._transfer_function(g_tilde)  # (B, n_units)
        logits = zH @ self.wLR.t()  # (B, n_classes)
        return logits

    # alias for torch-like API
    def forward(self, x: torch.Tensor):
        return self.predict(x)

    @staticmethod
    def _confidence_measure(zP: torch.Tensor, zT: torch.Tensor):
        """
        zP: (n_classes,) output activity vector (predicted)
        zT: (n_classes,) target population-coded vector (one-hot)
        returns: m (scalar) as in paper: positive when correct, negative when wrong
        """
        # compute first and second maximums
        vals, idx = torch.topk(zP, k=min(2, zP.numel()))
        u = vals[0] - (vals[1] if vals.size(0) > 1 else 0.0)
        if torch.argmax(zP) == torch.argmax(zT):
            return u.item()
        else:
            return -u.item()

    def _neighborhood(self, bmu_idx: int):
        """
        Returns a (n_units,) weight vector based on 2D distance from BMU in grid coords.
        Uses Gaussian on grid distances with current sigma.
        """
        bmu_coord = self.coords[bmu_idx].unsqueeze(0)  # (1,2)
        d2 = ((self.coords - bmu_coord) ** 2).sum(dim=1)  # (n_units,)
        nb = torch.exp(-d2 / (2.0 * (self.sigma**2)))  # (n_units,)
        return nb.unsqueeze(1)  # (n_units,1) for broadcasting

    def partial_fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Incremental update over a minibatch.
        X: (B, D) or (B, C, H, W) -> flattened inside
        Y: (B,) integer labels
        """
        if X.dim() > 2:
            X = X.view(X.size(0), -1)
        X = X.to(self.wSOM.device)
        Y = Y.to(self.wSOM.device).long()

        for i in range(X.size(0)):
            x = X[i].unsqueeze(0)  # (1, D)
            y = Y[i].item()
            # Hidden activations
            dists = self._distances(x)  # (1, n_units)
            g_tilde = self._gaussian(dists, self.kappa)  # (1, n_units)
            zH = self._transfer_function(g_tilde)  # (1, n_units)

            # Output prediction
            zP = (zH @ self.wLR.t()).squeeze(0)  # (n_classes,)
            # target population vector (one-hot)
            zT = torch.zeros(self.n_classes, device=self.wSOM.device)
            zT[y] = 1.0

            # confidence measure m
            m = self._confidence_measure(zP, zT)

            # gating lambda(t): train SOM when m <= theta (uncertain or wrong)
            lambda_mod = 1 if m <= self.theta else 0

            # update linear readout if there is sufficient hidden activity
            if zH.max() > self.theta:
                # \Delta wLR = 2 * epsLR * zH * (zP - zT)
                # simple online linear regression step (vectorized)
                pred = self.wLR @ zH.squeeze(0)  # (n_classes,)
                err = (pred - zT).unsqueeze(1)  # (n_classes, 1)
                # update with learning rate scaled by hidden activation
                self.wLR = self.wLR - self.lr_lr * (err @ zH)  # (n_classes, n_units)

            # update SOM prototypes locally if allowed by gating
            if lambda_mod:
                # BMU index
                d_flat = dists.view(-1)
                bmu = torch.argmin(d_flat).item()
                nb = self._neighborhood(bmu)  # (n_units, 1)
                delta = x - self.wSOM  # (n_units, D)
                self.wSOM = self.wSOM + self.lr_som * (nb * delta)

    def evaluate(self, loader):
        """
        Evaluate accuracy on a PyTorch DataLoader that yields (X, y).
        Returns accuracy in percent.
        """
        total = 0
        correct = 0
        self.eval()
        with torch.no_grad():
            for X, y in loader:
                if X.dim() > 2:
                    X = X.view(X.size(0), -1)
                logits = self.predict(X.to(self.wSOM.device))
                preds = logits.argmax(dim=1).cpu()
                correct += (preds == y).sum().item()
                total += y.size(0)
        self.train()
        return 100.0 * correct / max(1, total)
