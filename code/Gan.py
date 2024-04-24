import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MusicGAN:
    def __init__(self, device, epochs=50, batch_size=32):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(88 + 4, 256),  # Input: Concatenated noise vector and one-hot emotion labels
            nn.LeakyReLU(0.2),
            nn.Linear(256, 10000 * 88),  # Output to match the music data dimension
            nn.Tanh()
        ).to(device)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(10000 * 88 + 4, 512),  # Input: Concatenated music data and one-hot emotion labels
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ).to(device)

        # Loss and Optimizers
        self.loss_function = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.001, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))

    def train(self, music_data, label_data):
        dataset = TensorDataset(torch.tensor(music_data, dtype=torch.float), torch.tensor(label_data, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.epochs):
            for real_data, labels in loader:
                real_data = real_data.view(real_data.size(0), -1).to(self.device)
                labels = nn.functional.one_hot(labels - 1, num_classes=4).float().to(self.device)

                # Train Discriminator
                self.discriminator.zero_grad()
                real_output = self.discriminator(torch.cat((real_data, labels), 1))
                real_loss = self.loss_function(real_output, torch.ones(real_data.size(0), 1).to(self.device))

                noise = torch.randn(real_data.size(0), 88).to(self.device)
                fake_data = self.generator(torch.cat((noise, labels), 1))
                fake_output = self.discriminator(torch.cat((fake_data.detach(), labels), 1))
                fake_loss = self.loss_function(fake_output, torch.zeros(real_data.size(0), 1).to(self.device))
git
                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_d.step()

                # Train Generator
                self.generator.zero_grad()
                fake_output = self.discriminator(torch.cat((fake_data, labels), 1))
                g_loss = self.loss_function(fake_output, torch.ones(real_data.size(0), 1).to(self.device))
                g_loss.backward()
                self.optimizer_g.step()

            print(
                f'Epoch {epoch + 1}/{self.epochs}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}')

    def save_model(self, path):
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict()
        }, path)
