import numpy as np
import torch as T

class Trainer():
    def __init__(self, model, optimizer, loss_function, save_weight_path, logger = None) -> None:
        self.M = model
        self.O = optimizer
        self.L = loss_function
        self.save_path = save_weight_path

        self.train_losses = []
        self.current_epoch = 0
        self.logger = logger

    def __call__(self, dataloader, n_epochs = 20):
        self.current_epoch = 0
        losses = []
        epochs = []
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}/{n_epochs}")
            N = len(dataloader)
            epochs += [epoch+i/N for i in range(N)]
            
            epoch_losses = self._train_one_epoch(dataloader, self.M)
            losses += epoch_losses

            if (self.logger != None):
                pass

        self._save_model(self.save_path)
        return np.array(epochs), np.array(losses)

    def _train_one_epoch(self, dataloader, model):
        losses = []
        for i, (x, y) in enumerate(dataloader):
            self.O.zero_grad()
            loss = self.L(model(x), y)
            loss.backward()
            self.O.step()
            print(loss.item())

            losses.append(loss.item())
        return losses

    def _validate(self, dataloader):
        pass

    def _save_model(self, path):
        T.save(self.M.state_dict(), path)
