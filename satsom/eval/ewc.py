import torch
import torch.nn as nn
import torch.optim as optim


class OnlineEWC:
    """
    Online Elastic Weight Consolidation (Schwarz et al., 2018) maintaining a running estimate of parameter importance.
    """

    def __init__(self, model: nn.Module, device: str = "cpu", alpha: float = 0.9):
        self.model = model
        self.device = device
        # trainable parameters
        self.params = {
            n: p for n, p in self.model.named_parameters() if p.requires_grad
        }
        # initialize running fisher estimate and running means
        self._fisher = {
            n: torch.zeros_like(p, device=device) for n, p in self.params.items()
        }
        self._opt_mean = {
            n: torch.zeros_like(p, device=device) for n, p in self.params.items()
        }
        self.alpha = alpha
        self.steps = 0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        After a standard backward() call, update running Fisher and parameter means.
        x, y: single-sample tensors, shape [1,...] and [1]
        """
        # accumulate squared gradients
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grad2 = p.grad.data.clone().pow(2)
                self._fisher[name] = (
                    self.alpha * self._fisher[name] + (1 - self.alpha) * grad2
                )
                # running mean of parameters
                self._opt_mean[name] = (
                    self.alpha * self._opt_mean[name]
                    + (1 - self.alpha) * p.data.clone()
                )
        self.steps += 1

    def penalty(self, lam: float = 1e4) -> torch.Tensor:
        loss = 0
        for name, p in self.params.items():
            fisher = self._fisher[name]
            mean = self._opt_mean[name]
            loss += (fisher * (p - mean).pow(2)).sum()
        return lam * loss


class MLP_EWC(nn.Module):
    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_sizes: list = [400, 200],
        output_size: int = 10,
        device: str = "cpu",
        alpha: float = 0.9,
    ):
        super().__init__()
        self.device = device
        layers = []
        last = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, output_size))
        self.network = nn.Sequential(*layers).to(device)
        self.ewc = OnlineEWC(self, device, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        return self.network(x)

    def partial_fit(
        self,
        optimizer: optim.Optimizer,
        data: torch.Tensor,
        target: torch.Tensor,
        epochs: int = 1,
        lam: float = 1e4,
    ) -> float:
        """
        Online training with EWC penalty and Fisher updates.
        data: [batch_size, 1, 28,28] or [batch_size, 784], target: [batch_size]
        """
        self.train()
        losses = []
        for _ in range(epochs):
            for i in range(data.size(0)):
                optimizer.zero_grad()
                x = data[i].unsqueeze(0).to(self.device).view(1, -1)
                y = target[i].unsqueeze(0).to(self.device)
                out = self.forward(x)
                loss = nn.functional.cross_entropy(out, y)
                # backward and update Fisher
                loss.backward()
                # update EWC statistics
                self.ewc.update(x, y)
                # add penalty
                loss = loss + self.ewc.penalty(lam)
                # perform parameter update
                optimizer.step()
                losses.append(loss.item())
        return sum(losses) / len(losses)


class CNN_EWC(nn.Module):
    def __init__(self, output_size: int = 10, device: str = "cpu", alpha: float = 0.9):
        super().__init__()
        self.device = device
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ).to(device)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        ).to(device)
        self.ewc = OnlineEWC(self, device, alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        features = self.feature_extractor(x)
        return self.classifier(features)

    def partial_fit(
        self,
        optimizer: optim.Optimizer,
        data: torch.Tensor,
        target: torch.Tensor,
        epochs: int = 1,
        lam: float = 1e4,
    ) -> float:
        """
        Online training with EWC penalty and Fisher updates.
        data: [batch_size,1,28,28], target: [batch_size]
        """
        self.train()
        losses = []
        for _ in range(epochs):
            for i in range(data.size(0)):
                optimizer.zero_grad()
                x = data[i].unsqueeze(0).to(self.device)
                y = target[i].unsqueeze(0).to(self.device)
                out = self.forward(x)
                loss = nn.functional.cross_entropy(out, y)
                loss.backward()
                self.ewc.update(x, y)
                loss = loss + self.ewc.penalty(lam)
                optimizer.step()
                losses.append(loss.item())
        return sum(losses) / len(losses)
