import os
import torch
from torch.utils.data import Dataset, DataLoader
from model_1d import SupConClipResNet1d
from modeldecodeorth1 import Decoder, Decoder_dualNeuro
from multi_loss import SupConLoss, csLoss
from util import AverageMeter, save_model, TwoDropTransform_twomodal
import torch.nn as nn
import torch.optim as optim

# === Custom Dataset ===
class GpiStnDataset(Dataset):
    def __init__(self, gpi_tensor, stn_tensor, transform=None):
        assert gpi_tensor.shape[0] == stn_tensor.shape[0], "Mismatch in sample count"
        self.gpi = gpi_tensor
        self.stn = stn_tensor
        self.transform = transform

    def __len__(self):
        return self.gpi.shape[0]

    def __getitem__(self, idx):
        gpi_sample = self.gpi[idx]
        stn_sample = self.stn[idx]
        if self.transform:
            return self.transform(gpi_sample, stn_sample)
        return gpi_sample, stn_sample

# === Training Function ===
def train(loader, model_gpi, model_stn, model_decoder, mse_criterion, optimizer,
          optimizer_gpi, optimizer_stn, epoch, device):
    model_gpi.train()
    model_stn.train()
    model_decoder.train()

    for batch_idx, (gpi, stn) in enumerate(loader):
        gpi = gpi.to(device).float()
        stn = stn.to(device).float()

        optimizer.zero_grad()
        optimizer_gpi.zero_grad()
        optimizer_stn.zero_grad()

        features_gpi = model_gpi(gpi)
        features_stn = model_stn(stn)

        _, _, gpi_pred, stn_pred, _, _ = model_decoder(features_gpi, features_stn)

        assert gpi_pred.shape == gpi.shape, "Shape mismatch in GPi reconstruction"
        assert stn_pred.shape == stn.shape, "Shape mismatch in STN reconstruction"

        loss = mse_criterion(gpi_pred, gpi) + mse_criterion(stn_pred, stn)
        # loss = mse_criterion(stn_pred, stn)

        loss.backward()
        optimizer.step()
        optimizer_gpi.step()
        optimizer_stn.step()

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

# === Main Script ===
def main():
    subject_list = ["s508", "s514", "s515", "s517", "s519", "s520", "s521", "s523"]
    data_save_dir = r"F:\comp_project\Off_tensor_Data_R"
    model_save_dir = r"F:\comp_project\shared_ae_models"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for subj in subject_list:
        print(f"\n🚀 Processing subject {subj}...")

        # Load tensors
        save_dir = os.path.join(data_save_dir, subj)
        gpi = torch.load(os.path.join(save_dir, "gpi_train_off.pt"))  # [N, W, C]
        stn = torch.load(os.path.join(save_dir, "stn_train_off.pt"))  # [N, W, C]

        # Truncate to first 244 time points
        gpi = gpi[:, :244, :]  # shape: [N, 244, C]
        stn = stn[:, :244, :]  # shape: [N, 244, C]
        print(f"✅ Loaded: {gpi.shape}, {stn.shape}")

        # Reshape to [N, C, W]
        gpi = gpi.permute(0, 2, 1)
        stn = stn.permute(0, 2, 1)
        W = gpi.shape[-1]
        newW = W - 2

        # transform = TwoDropTransform_twomodal(W, newW)
        dataset = GpiStnDataset(gpi, stn, transform=None)
        loader = DataLoader(dataset, batch_size=8, shuffle=True) #default was 64

        # Define models
        model_gpi = SupConClipResNet1d(in_channel=gpi.shape[1], name1='resnet18', name2='resnet18', flag='neural', feat_dim=8).to(device) #I think this needs to change to 10
        model_stn = SupConClipResNet1d(in_channel=stn.shape[1], name1='resnet18', name2='resnet18', flag='neural', feat_dim=8).to(device)

        model_decoder = Decoder_dualNeuro(
            embedding_dim=8, # from encoder output
            channels=newW, # time dimension
            bottleneck_dim=8, # smaller internal representation
            image_latent_dim=3, #shared_GPi latent
            neural_latent_dim=3, #shared STN latent
            image_dim=gpi.shape[1], #GPi output channels
            neural_dim=stn.shape[1], #STN output channels
            image_private_dim=5, # GPi private
            neural_private_dim=5 #STN private
        ).to(device)

        # Define losses and optimizers
        mse_criterion = nn.MSELoss().to(device)
        optimizer_gpi = optim.Adam(model_gpi.parameters(), lr=1e-4)
        optimizer_stn = optim.Adam(model_stn.parameters(), lr=1e-4)
        optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=1e-4)

        # Train
        for epoch in range(1, 101):
            print(f"📚 Epoch {epoch}")
            train(loader, model_gpi, model_stn, model_decoder, mse_criterion,
                  optimizer_decoder, optimizer_gpi, optimizer_stn, epoch, device)

        # Save
        save_dir_model = os.path.join(model_save_dir, subj)
        os.makedirs(save_dir_model, exist_ok=True)
        save_model(model_gpi, optimizer_gpi, epoch, os.path.join(save_dir_model, "gpi_encoder.pth"))
        save_model(model_stn, optimizer_stn, epoch, os.path.join(save_dir_model, "stn_encoder.pth"))
        save_model(model_decoder, optimizer_decoder, epoch, os.path.join(save_dir_model, "decoder.pth"))

        print(f"✅ Finished training and saving model for {subj}.")

if __name__ == "__main__":
    main()
