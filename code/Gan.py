import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

class MusicGAN(nn.Module):
    def __init__(self, batch_size, epochs):
        super(MusicGAN, self).__init__()
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.criterion = nn.BCELoss()
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_generator(self):
        # The generator concatenates noise (100-dim) and a one-hot encoded emotion label (4-dim) as input
        model = nn.Sequential(
            nn.Linear(104, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1500),  # Output reshaped to [500, 3]  # Output range between 0 and 1
        )
        return model

    def build_discriminator(self):
        # The discriminator input is the flattened music array concatenated with the one-hot emotion label
        model = nn.Sequential(
            nn.Linear(1504, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        return model

    def forward(self, z, labels):
        return self.generator(torch.cat([z, labels], dim=1))

    def train(self, music_data, labels):
        self.to(self.device)
        # Convert lists to tensors
        music_data = torch.tensor(music_data, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(music_data, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        real_label = 0.9
        fake_label = 0.1
        for epoch in range(self.epochs):
            d_loss_total = 0.0
            g_loss_total = 0.0
            for i, (musics, emotion_labels) in enumerate(dataloader):
                musics = musics.to(self.device).view(musics.size(0), -1)
                emotion_labels = torch.nn.functional.one_hot(emotion_labels - 1, num_classes=4).float().to(self.device)
                batch_size = musics.size(0)

                # Correct labels to match the output dimensions of the discriminator
                real_targets = torch.full((batch_size, 1), real_label, device=self.device)
                fake_targets = torch.full((batch_size, 1), fake_label, device=self.device)

                # Train Discriminator
                self.optimizer_D.zero_grad()
                real_output = self.discriminator(torch.cat([musics, emotion_labels], dim=1))
                real_loss = self.criterion(real_output, real_targets)

                noise = torch.randn(batch_size, 100, device=self.device)
                fake_musics = self.forward(noise, emotion_labels)
                fake_output = self.discriminator(torch.cat([fake_musics.detach(), emotion_labels], dim=1))
                fake_loss = self.criterion(fake_output, fake_targets)

                d_loss = real_loss + fake_loss
                d_loss.backward()
                self.optimizer_D.step()
                d_loss_total += d_loss.item()

                # Train Generator
                self.optimizer_G.zero_grad()
                output = self.discriminator(torch.cat([fake_musics, emotion_labels], dim=1))
                g_loss = self.criterion(output, real_targets)
                g_loss.backward()
                self.optimizer_G.step()
                g_loss_total += g_loss.item()

            # Print after each epoch
            print(
                f'Epoch {epoch + 1}/{self.epochs}, D Loss: {d_loss_total / len(dataloader):.4f},'
                f' G Loss: {g_loss_total / len(dataloader):.4f}')

    def generate(self, label):
        self.generator.eval()
        noise = torch.randn(1, 100, device=self.device)
        label_vec = torch.nn.functional.one_hot(torch.tensor([label-1]), num_classes=4).float().to(self.device)
        with torch.no_grad():
            generated_musics = self.forward(noise, label_vec).view(500, 3)
            # # Rescale each feature to its appropriate range
            # generated_musics[:, 0] = generated_musics[:, 0] * 87 + 21   # Scale notes
            # generated_musics[:, 1] = generated_musics[:, 1] * 127       # Scale velocities
            # generated_musics[:, 2] = generated_musics[:, 2] * 3000       # Scale time delays
            generated_musics = generated_musics.cpu().numpy()
        return generated_musics.astype(int)
