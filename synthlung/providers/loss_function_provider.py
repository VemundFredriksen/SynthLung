import torch.nn as nn

class LossFunctionProvider():
    def __init__(self, config):
        if (config["loss"] == "CE"):
            self.criterion = nn.CrossEntropyLoss()

    def __call__(self):
        return self.criterion