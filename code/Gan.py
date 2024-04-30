import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class MusicGAN:
    def __init__(self,  align, device, epochs=50, batch_size=32):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.align = align

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(100 + 4, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self.align * 88),
            nn.Tanh()
        ).to(self.device)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.align * 88 + 4, 512),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output a probability between 0 and 1
        ).to(self.device)

        # Loss and Optimizers
        self.loss_function = nn.BCELoss()
        self.optimizer_g = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_d = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

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

                noise = torch.randn(real_data.size(0), 100).to(self.device)
                fake_data = self.generator(torch.cat((noise, labels), 1))
                fake_output = self.discriminator(torch.cat((fake_data.detach(), labels), 1))
                fake_loss = self.loss_function(fake_output, torch.zeros(real_data.size(0), 1).to(self.device))

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

    def generate(self, emotion_label, device='cuda'):
        # Ensure the generator is in evaluation mode
        self.generator.eval()
        noise = torch.randn(1, 100, device=device)  # Adjust size if needed to match the first layer of the generator
        emotion = nn.functional.one_hot(torch.tensor([emotion_label - 1]), num_classes=4).float().to(device)
        gen_input = torch.cat((noise, emotion), 1)
        # Generate music data
        with torch.no_grad():
            generated_music = self.generator(gen_input)
        generated_music = generated_music.view(self.align, 88)
        generated_music = ((generated_music + 1) / 2) * 127  # Scale and shift
        threshold = 50  # Threshold can be adjusted based on desired sparsity
        generated_music = torch.where(generated_music < threshold, torch.zeros_like(generated_music), generated_music)

        return generated_music.cpu().numpy()