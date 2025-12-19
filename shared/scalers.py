import torch

class StandardScaler:
    def __init__(self, eps=1e-8):
        self.mean = None
        self.std = None
        self.eps = eps

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)
        return self

    def transform(self, x: torch.Tensor):
        return (x - self.mean) / (self.std + self.eps)

    def inverse_transform(self, x: torch.Tensor):
        return x * (self.std + self.eps) + self.mean

    def state_dict(self):
        return {
            "mean": self.mean,
            "std": self.std,
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state["eps"]



class MinMaxScaler:
    def __init__(self, feature_range=(-1.0, 1.0), eps=1e-8):
        self.min = None
        self.max = None
        self.low, self.high = feature_range
        self.eps = eps

    def fit(self, x: torch.Tensor):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values
        return self

    def transform(self, x: torch.Tensor):
        x_std = (x - self.min) / (self.max - self.min + self.eps)
        return x_std * (self.high - self.low) + self.low

    def inverse_transform(self, x: torch.Tensor):
        x_std = (x - self.low) / (self.high - self.low)
        return x_std * (self.max - self.min + self.eps) + self.min

    def state_dict(self):
        return {
            "min": self.min,
            "max": self.max,
            "feature_range": (self.low, self.high),
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.min = state["min"]
        self.max = state["max"]
        self.low, self.high = state["feature_range"]
        self.eps = state["eps"]
