import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


class MusicCGAN:
    def __init__(self, batch_size, epochs, align):
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.align = align

        # Generator
        self.generator = nn.Sequential(
            nn.Linear(100 + 4, 256),  # 100 noise dimensions, 4 one-hot for emotions
            nn.ReLU(),
            nn.Linear(256, self.align * 88),
            nn.Sigmoid()
        ).to(self.device)

        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Linear(self.align * 88 + 4, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.loss_fn = nn.BCELoss()

    def train(self, music_data, labels):
        dataset = TensorDataset(music_data, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        real_label = 1
        fake_label = 0

        for epoch in range(self.epochs):
            for i, (music, emotions) in enumerate(dataloader):
                music = music.to(self.device).view(-1, 15000 * 88)
                emotions = nn.functional.one_hot(emotions - 1, num_classes=4).float().to(self.device)

                # Train Discriminator
                self.discriminator.zero_grad()
                real_data = torch.cat((music, emotions), 1)
                output_real = self.discriminator(real_data)
                loss_real = self.loss_fn(output_real, torch.full((music.size(0),), real_label, device=self.device))

                noise = torch.randn(music.size(0), 100, device=self.device)
                fake_music = self.generator(torch.cat((noise, emotions), 1))
                fake_data = torch.cat((fake_music, emotions), 1)
                output_fake = self.discriminator(fake_data.detach())
                loss_fake = self.loss_fn(output_fake, torch.full((music.size(0),), fake_label, device=self.device))

                loss_D = (loss_real + loss_fake) / 2
                loss_D.backward()
                self.optimizer_D.step()

                # Train Generator
                self.generator.zero_grad()
                output_fake = self.discriminator(fake_data)
                loss_G = self.loss_fn(output_fake, torch.full((music.size(0),), real_label, device=self.device))
                loss_G.backward()
                self.optimizer_G.step()

                if (i + 1) % 50 == 0:
                    print(
                        f"Epoch [{epoch + 1}/{self.epochs}], Step [{i + 1}/{len(dataloader)}], Loss_D: {loss_D.item()}, Loss_G: {loss_G.item()}")

    def generate(self, emotion_label):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(1, 100, device=self.device)
            emotion = nn.functional.one_hot(torch.tensor([emotion_label - 1]), num_classes=4).float().to(self.device)
            generated_music = self.generator(torch.cat((noise, emotion), 1)).view(15000, 88)
            generated_music = generated_music * 127  # Scale output to range 0-127
            generated_music = torch.where(generated_music < 30, torch.zeros_like(generated_music), generated_music)
        return generated_music.cpu().numpy()

# Example Usage
# Assuming music_data_tensor and labels_tensor are loaded correctly with shapes [(N, 15000, 88), (N,)]
# cg = MusicCGAN(batch_size=32, epochs=10)
# cg.train(music_data_tensor, labels_tensor)
# generated_music = cg.generate(emotion_label=1)
