import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
'''nn.Linear()里具体多少要看music array是多少，我按照lxy给的图放的'''
'''还没train，但具体框架在这了'''
'''用了Pytorch，试试新的，正好我yolo那里也是pytorch'''
'''最好装一下cude support pytorch可以用GPU，否则CPU负荷太大，怎么装or更新群里问wzy'''


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


# GAN Class
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
        self.generator.to(device)
        self.discriminator.to(device)

        for epoch in range(epochs):
            for midi_arr, emo_labels in dataloader:
                midi_arr, emo_labels = midi_arr.to(device), emo_labels.to(device).squeeze()

                # Discriminator Training
                real_labels = torch.ones(midi_arr.size(0), 1, device=device)
                fake_labels = torch.zeros(midi_arr.size(0), 1, device=device)

                self.discriminator.zero_grad()
                real_outputs = self.discriminator(torch.cat([midi_arr, self.label_emb(emo_labels)], dim=1))
                d_loss_real = self.criterion(real_outputs, real_labels)

                noise = torch.randn(midi_arr.size(0), self.noise_dim, device=device)
                fake_midi = self.generator(torch.cat([noise, self.label_emb(emo_labels)], dim=1))
                fake_outputs = self.discriminator(torch.cat([fake_midi, self.label_emb(emo_labels)], dim=1))
                d_loss_fake = self.criterion(fake_outputs, fake_labels)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.optim_D.step()

                # Generator Training
                self.generator.zero_grad()
                reclassified_fake_outputs = self.discriminator(
                    torch.cat([fake_midi, self.label_emb(emo_labels)], dim=1))
                g_loss = self.criterion(reclassified_fake_outputs, real_labels)
                g_loss.backward()
                self.optim_G.step()

            print(f'Epoch {epoch + 1}: D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    def generate_music(self, emotion_label, num_samples=1):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generator.eval()
        noise = torch.randn(num_samples, self.noise_dim, device=device)
        labels = torch.LongTensor([emotion_label] * num_samples).to(device)
        with torch.no_grad():
            generated_music = self.generator(torch.cat([noise, self.label_emb(labels)], dim=1)).cpu().numpy()
        return generated_music

    @staticmethod
    def plot_generated_music(generated_music):
        plt.figure(figsize=(12, 6))
        plt.imshow(generated_music, aspect='auto', cmap='binary')
        plt.colorbar()
        plt.title('Generated Music Visual Representation')
        plt.xlabel('Time Steps')
        plt.ylabel('Piano Keys')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Placeholder for dataset loading logic
    # data_loader = DataLoader(...)
    # gan = EmotionMusicGAN().to(device)
    # gan.train(data_loader)
    pass