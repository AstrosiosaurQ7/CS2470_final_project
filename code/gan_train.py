from gan import MusicGAN
import torch
from torch.utils.data import DataLoader, TensorDataset
import json
import os
from midi_array import *


def write_music_data():
    label_general_path = '../data/EMOPIA/label.csv'
    folder_general_path = '../data/EMOPIA/midis'

    music_data, label_data = get_music_data(folder_general_path, label_general_path)
    return music_data, label_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Assuming `write_music_data` is a function that provides `music_data` and `label_data`
    music_data, label_data = write_music_data()
    gan = MusicGAN(device=device, epochs=20, batch_size=16)
    gan.train(music_data, label_data)
    gan.save_model("music_gan_model.pth")


if __name__ == "__main__":
    main()
