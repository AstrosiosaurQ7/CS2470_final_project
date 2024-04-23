import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from midi_array import *
import numpy as np
import os
import argparse


class MusicDataset(Dataset):
    def __init__(self, music_data):
        self.music_data = music_data

    def __len__(self):
        return len(self.music_data)

    def __getitem__(self, idx):
        item = self.music_data[idx]
        midi_arr = torch.FloatTensor(item['midi_arr'])
        emo_label = torch.LongTensor([int(item['emo_label']) - 1])
        return midi_arr, emo_label


class MusicGAN:
    def __init__(self, midi_dim, noise_dim=100, emotion_dim=10, batch_size=32, lr=0.0002, beta1=0.5):
        super().__init__()
        self.midi_dim = midi_dim
        self.noise_dim = noise_dim
        self.emotion_dim = emotion_dim
        self.batch_size = batch_size

        # Generator
        self.generator = nn.Sequential(
            nn.Embedding(4, emotion_dim),
            nn.Linear(noise_dim + emotion_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, midi_dim),
            nn.Tanh()
        )

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Embedding(4, emotion_dim),
            nn.Linear(midi_dim + emotion_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.label_emb = nn.Embedding(4, emotion_dim)
        self.optim_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optim_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.criterion = nn.BCELoss()

    def train(self, dataset, epochs):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        self.generator.to(device)
        self.discriminator.to(device)

        for epoch in range(epochs):
            for midi_arr, emo_labels in dataloader:
                real_data = midi_arr.to(device)
                labels = emo_labels.to(device)
                batch_size = real_data.size(0)

                # Train Discriminator
                self.discriminator.zero_grad()
                real_labels = torch.ones(batch_size, 1, device=device)
                fake_labels = torch.zeros(batch_size, 1, device=device)

                # Real data
                label_emb = self.discriminator[0](labels).view(batch_size, -1)
                real_inputs = torch.cat([real_data, label_emb], 1)
                output_real = self.discriminator(real_inputs)
                loss_real = self.criterion(output_real, real_labels)

                # Generated data
                noise = torch.randn(batch_size, self.noise_dim, device=device)
                fake_data = self.generator(torch.cat([noise, self.generator[0](labels).view(batch_size, -1)], 1))
                label_emb = self.discriminator[0](labels).view(batch_size, -1)
                fake_inputs = torch.cat([fake_data, label_emb], 1)
                output_fake = self.discriminator(fake_inputs)
                loss_fake = self.criterion(output_fake, fake_labels)

                # Backprop and optimize
                d_loss = (loss_real + loss_fake) / 2
                d_loss.backward()
                self.optim_D.step()

                # Train Generator
                self.generator.zero_grad()
                label_emb = self.generator[0](labels).view(batch_size, -1)
                generated_data = self.generator(torch.cat([noise, label_emb], 1))
                label_emb = self.discriminator[0](labels).view(batch_size, -1)
                gen_inputs = torch.cat([generated_data, label_emb], 1)
                output_gen = self.discriminator(gen_inputs)
                g_loss = self.criterion(output_gen, real_labels)

                g_loss.backward()
                self.optim_G.step()

            print(f'Epoch [{epoch + 1}/{epochs}] - Loss D: {d_loss.item()}, Loss G: {g_loss.item()}')

    def generate_music(self, emotion_label, num_samples=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator.eval()
        noise = torch.randn(num_samples, self.noise_dim, device=device)
        labels = torch.LongTensor([emotion_label] * num_samples).to(device)
        with torch.no_grad():
            label_emb = self.generator[0](labels).view(num_samples, -1)
            music = self.generator(torch.cat([noise, label_emb], 1)).detach().numpy()
        return music

    @staticmethod
    def plot_music(self, music):
        plt.figure(figsize=(12, 8))
        plt.imshow(music, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.title('Generated Music Visualization')
        plt.xlabel('Piano Keys')
        plt.ylabel('Time Steps')
        plt.show()


# Deprecate
if __name__ == "__main__":
    pass
