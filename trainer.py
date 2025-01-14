import torch

class Trainer():
    def __init__(self) -> None:
        pass        

    def train_epoch():
        pass

    def val_epoch():
        pass

    def test_epoch():
        pass

    def save_checkpoint():
        pass

    def load_checkpoint():
        pass

    def train():
        pass

    def val():
        pass

    def test():
        pass

    def fit(self, model, train_loader, val_loader, train_sampler):
        # Step 1: Configure the optimizer, mixed precision, learning rate scheduler
        optimizer = torch.optim.AdamW(model.parameters, lr=1e-4, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler()
        

        # Step 2: 



