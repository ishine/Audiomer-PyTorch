import torchmetrics
import torch
import gc
import numpy as np
from abc import ABC, abstractmethod


class MetricsBase(ABC):
    def __init__(self, num_classes, metrics_dict):
        self.num_classes = num_classes
        self.subset = metrics_dict

    @abstractmethod
    def __call__(self, y_pred, y, subset):
        pass

    def to(self, device):
        self.device = device
        for s in self.subset.keys():
            for m in self.subset[s].keys():
                self.subset[s][m].to(device)

    def reset(self, subset):
        for m in self.subset[subset].keys():
            self.subset[subset][m].reset()
        gc.collect()


class SpeechCommandsMetrics(MetricsBase):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.subset = {
            "train": {},
            "test": {},
            "val": {}
        }

        for s in self.subset.keys():
            self.subset[s]['ACC'] = torchmetrics.Accuracy(
                num_classes=self.num_classes, compute_on_step=False)

        super().__init__(num_classes, self.subset)

    def __call__(self, y_pred, y, subset):
        y_pred = y_pred.argmax(-1)
        if subset == "train":        
            try:
                self.subset[subset]["ACC"].update(y_pred, y)
            except:
                pass
        else:
            self.subset[subset]["ACC"].update(y_pred, y)

        m = dict()
        m["ACC"] = self.subset[subset]["ACC"].compute()
        return m

