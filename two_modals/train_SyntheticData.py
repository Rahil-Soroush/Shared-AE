import os
import torch
from torch.utils.data import Dataset, DataLoader
from model_1d import SupConClipResNet1d
from modeldecodeorth1 import  Decoder_dualNeuro
from multi_loss import SupConLoss, csLoss
from util import AverageMeter, save_model, TwoDropTransform_twomodal
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

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
def train(loader, model_gpi, model_stn, model_decoder, mse_criterion,cs_criterion, optimizer,
          optimizer_gpi, optimizer_stn, epoch, device, writer):
    model_gpi.train()
    model_stn.train()
    model_decoder.train()

    total_loss_epoch = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss3 = 0
    total_loss4 = 0
    total_loss5 = 0
    total_loss6 = 0
    n_batches = 0


    for batch_idx, (gpi, stn) in enumerate(loader):
        gpi = gpi.to(device).float()
        stn = stn.to(device).float()

        optimizer.zero_grad()
        optimizer_gpi.zero_grad()
        optimizer_stn.zero_grad()

        features_gpi = model_gpi(gpi)
        features_stn = model_stn(stn)

        shared_gpi,shared_stn, gpi_pred, stn_pred, gpi_prv, stn_prv = model_decoder(features_gpi, features_stn)

        loss1 = mse_criterion(gpi_pred, gpi)
        loss2 = mse_criterion(stn_pred, stn)
        loss3=cs_criterion(shared_gpi,shared_stn)
        loss4=cs_criterion(gpi_prv,stn_prv) #inverse
        loss5=cs_criterion(shared_gpi,gpi_prv) #inverse
        loss6=cs_criterion(shared_stn,stn_prv) #inverse

        # Final loss with weights
        weight3 = 1.0      # shared/shared alignment
        # weight4 = 0.05     # private/private
        # weight5 = 0.05     # shared/private
        weight4 = 0.01     # private/private
        weight5 = 0.01     # shared/private

        loss = (
            loss1 * gpi.shape[1] * gpi.shape[2] +   # N_channels * T
            loss2 * stn.shape[1] * stn.shape[2] +
            weight3 * loss3 +
            weight4 / loss4 +
            weight5 / loss5 +
            weight5 / loss6
        )
        total_loss_epoch += loss.item()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        total_loss3 += loss3.item()
        total_loss4 += loss4.item()
        total_loss5 += loss5.item()
        total_loss6 += loss6.item()
        n_batches += 1

        
        loss.backward()
        optimizer.step()
        optimizer_gpi.step()
        optimizer_stn.step()

        # # Log each loss component to TensorBoard
        # step = epoch * len(loader) + batch_idx
        # writer.add_scalar("Loss/total", loss.item(), step)
        # writer.add_scalar("Loss/recon_gpi", loss1.item(), step)
        # writer.add_scalar("Loss/recon_stn", loss2.item(), step)
        # writer.add_scalar("Loss/shared_cs", loss3.item(), step)
        # writer.add_scalar("Loss/private_private_inv_cs", loss4.item(), step)
        # writer.add_scalar("Loss/shared_gpi_private_inv_cs", loss5.item(), step)
        # writer.add_scalar("Loss/shared_stn_private_inv_cs", loss6.item(), step)

        if batch_idx % 20 == 0:
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}")

    writer.add_scalar("Loss/total", total_loss_epoch / n_batches, epoch)
    writer.add_scalar("Loss/recon_gpi", total_loss1 / n_batches, epoch)
    writer.add_scalar("Loss/recon_stn", total_loss2 / n_batches, epoch)
    writer.add_scalar("Loss/shared_cs", total_loss3 / n_batches, epoch)
    writer.add_scalar("Loss/private_inv_cs", total_loss4 / n_batches, epoch)
    writer.add_scalar("Loss/gpi_sh_p_inv_cs", total_loss5 / n_batches, epoch)
    writer.add_scalar("Loss/stn_sh_p_inv_cs", total_loss6 / n_batches, epoch)


# === Main Script ===
def main():

    data_dir = r"F:\comp_project\synthecticData\data"
    model_save_dir = r"F:\comp_project\synthecticData\SharedAE"
    writer = SummaryWriter(log_dir=f"runs/sharedae_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    print(f"\n🚀 Processing ")

    # Load tensors
    # save_dir = os.path.join(data_save_dir, subj)
    data = np.load(os.path.join(data_dir, "synth_data_v4_lin_noDelay_morePri_L250_fs500_s3p3.npz"))  # [N, W, C]
    region1 = data['region1']           # shape: (n_trials, n_channels, T)
    region2 = data['region2']           # shape: (n_trials, n_channels, T)
    gt_shared = data['gt_shared']       # shape: (n_trials, shared_dim, T)
    gt_shared1 = data['gt_shared1']     # shape: (n_trials, shared_dim, T)
    gt_shared2 = data['gt_shared2']     # shape: (n_trials, shared_dim, T)
    gt_private1 = data['gt_private1']   # shape: (n_trials, private_dim, T)
    gt_private2 = data['gt_private2']   # shape: (n_trials, private_dim, T)

    # Truncate to first 244 time points
    gpi = region1[:, :, :244]  # shape: [N, C, 244]
    stn = region2[:, :, :244]  
    print(f"✅ Loaded: {gpi.shape}, {stn.shape}")

    # Reshape to [N, C, W]
    # gpi = gpi.permute(0, 2, 1)
    # stn = stn.permute(0, 2, 1)
    W = gpi.shape[-1]
    newW = W - 2

    # transform = TwoDropTransform_twomodal(W, newW)
    dataset = GpiStnDataset(gpi, stn, transform=None)
    loader = DataLoader(dataset, batch_size=8, shuffle=True) #default was 64

    # Define models
    model_gpi = SupConClipResNet1d(in_channel=gpi.shape[1], name1='resnet18', name2='resnet18', flag='neural', feat_dim=10).to(device) #I think this needs to change to 10
    model_stn = SupConClipResNet1d(in_channel=stn.shape[1], name1='resnet18', name2='resnet18', flag='neural', feat_dim=10).to(device)

    model_decoder = Decoder_dualNeuro(
        embedding_dim=10, # from encoder output
        channels=newW, # time dimension
        bottleneck_dim=10, # smaller internal representation
        image_latent_dim=3, #shared_GPi latent
        neural_latent_dim=3, #shared STN latent
        image_dim=gpi.shape[1], #GPi output channels
        neural_dim=stn.shape[1], #STN output channels
        image_private_dim=3, # GPi private
        neural_private_dim=3 #STN private
    ).to(device)

    # Define losses and optimizers
    mse_criterion = nn.MSELoss().to(device)
    cs_criterion=csLoss(15).to(device)

    optimizer_gpi = optim.Adam(model_gpi.parameters(), lr=1e-4)
    optimizer_stn = optim.Adam(model_stn.parameters(), lr=1e-4)
    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=1e-4)

    # Train
    for epoch in range(1, 201):
        print(f"📚 Epoch {epoch}")
        train(loader, model_gpi, model_stn, model_decoder, mse_criterion,cs_criterion,
                optimizer_decoder, optimizer_gpi, optimizer_stn, epoch, device, writer)

    writer.close()
    # Save
    save_dir_model = os.path.join(model_save_dir, "synth_data_v4_lin_noDelay_morePri_L250_fs500_s3p3")
    os.makedirs(save_dir_model, exist_ok=True)
    save_model(model_gpi, optimizer_gpi, epoch, os.path.join(save_dir_model, "gpi_encoder.pth"))
    save_model(model_stn, optimizer_stn, epoch, os.path.join(save_dir_model, "stn_encoder.pth"))
    save_model(model_decoder, optimizer_decoder, epoch, os.path.join(save_dir_model, "decoder.pth"))

    print(f"✅ Finished training and saving model.")

if __name__ == "__main__":
    main()
