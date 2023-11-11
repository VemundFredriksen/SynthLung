import torch.optim as optim

class OptimizerProvider():
    def __init__(self, config):
        self.lr = config["learning_rate"]
        if (config["optimizer"] == "Adam"):
            self.optimizer = optim.Adam

    def __call__(self, model):
        return self.optimizer(model.parameters(), lr=self.lr)