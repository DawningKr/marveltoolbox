from abc import ABC, abstractmethod
from typing import Any


class LoRA(ABC):
    @abstractmethod
    def get_lora_layers(self) -> Any:
        pass


class LoRAConfig(ABC):
    def __init__(self):
        self.rank = None
        self.alpha = None
        self.lora_A = None
        self.lora_B = None
        self.scale = None
        self.enable_lora = False

    @abstractmethod
    def set_lora_configs(self, rank, alpha):
        self.rank = rank
        self.alpha = alpha
        self.scale = rank / alpha

    def set_lora_status(self, enable_lora: bool):
        self.enable_lora = enable_lora
        self._set_params_status(enable_lora)

    @abstractmethod
    def _set_params_status(self, freeze_params: bool):
        pass
