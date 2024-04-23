import numpy as np
from Gan import MusicGAN, MusicDataset
import torch
from torch.utils.data import DataLoader
import json
import os


def main():
    music_data = []
    files = os.listdir('./test')
    index = 0
    for filename in files:
        # Get file info
        with open(f'./test/{filename}', 'r') as file:
            temp_dict = json.load(file)
            music_data.append({
                'midi_arr': temp_dict['midi_arr'],
                'emo_label': temp_dict['emo_label']
            })

    # batch size
    batch_size = 4

    # DataLoader setup
    dataset = MusicDataset(music_data)
    midi_dim = len(music_data[0]['midi_arr'][0]) * len(music_data[0]['midi_arr'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Device and size selection
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5

    music_gan = MusicGAN(midi_dim=midi_dim, batch_size=batch_size)
    music_gan.train(data_loader, epochs=epochs)
    return


if __name__ == "__main__":
    main()
