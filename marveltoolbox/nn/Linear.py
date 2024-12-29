from torch import nn
from ..LoRA import LoRAConfig


class Linear(nn.Module, LoRAConfig):
    def __init__(self, in_features, out_features) -> None:
        nn.Module.__init__(self)
        LoRAConfig.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(self.in_features, self.out_features)

    def forward(self, X):
        if self.enable_lora:
            return self.linear(X) + self.lora_B(self.lora_A(X)) * self.scale
        else:
            return self.linear(X)

    def set_lora_configs(self, rank, alpha):
        LoRAConfig.set_lora_configs(self, rank, alpha)
        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        nn.init.normal_(self.lora_A.weight)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)
        nn.init.zeros_(self.lora_B.weight)

    def _set_params_status(self, freeze_params: bool):
        for param in self.linear.parameters():
            param.requires_grad = not freeze_params


