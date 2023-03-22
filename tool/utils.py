import numpy as np
import torch


class EarlyStopping:
    """
    Early stops the training if loss doesn't improve after a given patience.
    """
    def __init__(self, patience=80):

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.loss_min = np.Inf

    def __call__(self, loss):
        if np.isnan(loss):
            self.early_stop = True
        score = -loss

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0