import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import UnderwaterDataset, data_prep
from model import UNet
from loss import MS_SSIM_L1_LOSS
from TRAINING_CONFIG import *

def setup():
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = MS_SSIM_L1_LOSS()
    return model, opt, loss

def train(train_dataloader, val_dataloader, save_model=False):
    model, opt, loss = setup()

    for epoch in range(num_epochs):
        for sample in tqdm(train_dataloader, desc=f'[Train]', leave=False):
            inp, label = sample['input'].to(device), sample['label'].to(device)
            opt.zero_grad()
            output = model(inp)
            error = loss(output, label)
            error.backward()
            opt.step()

        if save_model and (epoch + 1) % snapshot_freq == 0:
            torch.save(model.state_dict(), f"{snapshots_folder}model_epoch_{epoch}_{model_name}.ckpt")

def run_training():
    print("Device is set to", device)
    train_dataloader, val_dataloader = data_prep()
    train(train_dataloader, val_dataloader, save_model=True)

if __name__ == '__main__':
    run_training()
