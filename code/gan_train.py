import numpy as np
from gan import MusicGAN, MusicDataset
import torch
from torch.utils.data import DataLoader
import json
import os
from midi_array import *


def main():
    # TODO choose correct system path
    label_general_path = '..-data-EMOPIA-label.csv'
    folder_general_path = '..-data-EMOPIA-midis'

    label_path_win = label_general_path.replace('-', '/')
    folder_path_win = folder_general_path.replace('-', '/')
    label_path_mac = label_general_path.replace('-', '\\')
    folder_path_mac = folder_general_path.replace('-', '\\')

    music_data = get_music_data(folder_path_win, label_path_win)

    # batch size
    batch_size = 64

    # DataLoader setup
    dataset = MusicDataset(music_data)
    midi_dim = len(music_data[0]['midi_arr'][0]) * len(music_data[0]['midi_arr'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Device and size selection
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 50

    music_gan = MusicGAN(midi_dim=midi_dim, batch_size=batch_size)
    music_gan.train(data_loader, epochs=epochs)
    return


if __name__ == "__main__":
    main()
