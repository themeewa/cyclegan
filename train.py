import torch
from dataset import NoFireFireDataset
import sys
import config
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint
from discriminator_model import Discriminator
from generator_model import Generator
from torchvision.utils import save_image
from tqdm import tqdm

def train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(loader, leave=True)

    for idx, (fire, nofire) in enumerate(loop):
        fire = fire.to(config.DEVICE)
        nofire = nofire.to(config.DEVICE)

        # Train Discriminators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            fake_nofire = gen_H(fire)
            D_H_real = disc_H(nofire)
            D_H_fake = disc_H(fake_nofire.detach())
            D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
            D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
            D_H_loss = (D_H_real_loss + D_H_fake_loss) / 2

            fake_fire = gen_Z(nofire)
            D_Z_real = disc_Z(fire)
            D_Z_fake = disc_Z(fake_fire.detach())
            D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
            D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
            D_Z_loss = (D_Z_real_loss + D_Z_fake_loss) / 2

            D_loss = (D_H_loss + D_Z_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators H and Z
        with torch.amp.autocast(device_type=config.DEVICE):
            D_H_fake = disc_H(fake_nofire)
            D_Z_fake = disc_Z(fake_fire)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

            cycle_fire = gen_Z(fake_nofire)
            cycle_nofire = gen_H(fake_fire)
            cycle_fire_loss = l1(fire, cycle_fire)
            cycle_nofire_loss = l1(nofire, cycle_nofire)

            identity_nofire = gen_H(nofire)
            identity_fire = gen_Z(fire)
            identity_nofire_loss = l1(nofire, identity_nofire)
            identity_fire_loss = l1(fire, identity_fire)

            G_loss = (
                loss_G_Z
                + loss_G_H
                + cycle_nofire_loss * config.LAMBDA_CYCLE
                + cycle_fire_loss * config.LAMBDA_CYCLE
                + identity_nofire_loss * config.LAMBDA_IDENTITY
                + identity_fire_loss * config.LAMBDA_IDENTITY
            )
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % config.SAVE_FREQ == 0:
            save_image(
                fake_nofire,
                f"saved_images/nofire_{idx}.png",
                normalize=True,
            )
            save_image(
                fake_fire,
                f"saved_images/fire_{idx}.png",
                normalize=True,
            )
        

def main():
    disc_H = Discriminator(in_channels=3).to(config.DEVICE)
    disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
    gen_Z = Generator(img_channel=3, num_residuals=9).to(config.DEVICE)
    gen_H = Generator(img_channel=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H, gen_H, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z, gen_Z, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H, disc_H, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z, disc_Z, opt_disc, config.LEARNING_RATE,
        )

    dataset = NoFireFireDataset(
        root_nofire=config.TRAIN_DIR + "/nofire",
        root_fire=config.TRAIN_DIR + "/fire",
        transform=config.transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # g_scaler = torch.cuda.amp.GradScaler()
    g_scaler = torch.GradScaler(device=config.DEVICE)
    # d_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.GradScaler(device=config.DEVICE) # For MPS

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc_H, disc_Z, gen_Z, gen_H, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler)
        print(f"Epoch [{epoch}/{config.NUM_EPOCHS}] completed.")
        if config.SAVE_MODEL:
            save_checkpoint(gen_H, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_Z, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_H, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_Z, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

if __name__ == "__main__":
    main()