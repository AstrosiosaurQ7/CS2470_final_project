import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse
'''nn.Linear()里具体多少要看music array是多少，我按照lxy给的图放的'''
'''还没train，但具体框架在这了'''
'''用了Pytorch，试试新的，正好我yolo那里也是pytorch'''
'''最好装一下cude support pytorch可以用GPU，否则CPU负荷太大，怎么装or更新群里问wzy'''

class MusicGAN(nn.Module):
    def __init__(self):
        super(MusicGAN, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 29617 * 88),
            nn.Sigmoid()  # Output values between 0 and 1
        )

        self.discriminator = nn.Sequential(
            nn.Linear(29617 * 88, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Output values between 0 and 1
        )

    def preprocess(self, data):
        # TODO
        # data process
        return data

    def generator_loss(self, generated_data):
        gen_labels = torch.ones(generated_data.size(0), 1, device=generated_data.device)
        return nn.BCELoss()(self.discriminator(generated_data), gen_labels)

    def discriminator_loss(self, real_data, generated_data):
        real_labels = torch.ones(real_data.size(0), 1, device=real_data.device)
        real_loss = nn.BCELoss()(self.discriminator(real_data), real_labels)

        gen_labels = torch.zeros(generated_data.size(0), 1, device=generated_data.device)
        fake_loss = nn.BCELoss()(self.discriminator(generated_data), gen_labels)

        return real_loss + fake_loss

    '''choose train device'''
    def train(self, data_loader, epochs=100, lr=0.002, device=torch.device('cpu')):
        optimizer_gen = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_disc = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        d_loss, g_loss = {}, {}
        for epoch in range(epochs):
            for real_data in data_loader:
                real_data = real_data.to(device).view(real_data.size(0), -1)  # Flatten the data
                batch_size = real_data.size(0)

                # Train discriminator
                z = torch.randn(batch_size, 100, device=device)
                generated_data = self.generator(z)
                d_loss = self.discriminator_loss(real_data, generated_data)
                optimizer_disc.zero_grad()
                d_loss.backward()
                optimizer_disc.step()

                # Train generator
                z = torch.randn(batch_size, 100, device=device)
                generated_data = self.generator(z)
                g_loss = self.generator_loss(generated_data)
                optimizer_gen.zero_grad()
                g_loss.backward()
                optimizer_gen.step()

            print(f'Epoch {epoch + 1}/{epochs}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}')


if __name__ == "__main__":
    # Simulate data assuming 500 pieces, each is [29617, 88]
    # TODO
    # import data here
    music_data = torch.randint(0, 2, (500, 29617, 88), dtype=torch.float32)

    # batch size
    batch_size = 100

    # DataLoader setup
    dataset = TensorDataset(music_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model initialization and training
    gan = MusicGAN().to(device)
    gan.train(data_loader, epochs=5, device=device)
    # TODO
    # evaluate and test
