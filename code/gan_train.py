import numpy as np

from gan import MusicGAN
import torch
from midi_array import *
import json

def write_music_data():
    label_general_path = '../data/EMOPIA/label.csv'
    folder_general_path = '../data/EMOPIA/midis'
    return get_music_data(folder_general_path, label_general_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(torch.cuda.get_device_name())
    music_data, label_data, align = write_music_data()
    print(music_data[0][300])
    # print("midi done")
    # gan = MusicGAN(align, device=device, epochs=10, batch_size=64)
    # gan.train(music_data, label_data)
    # # generate in 1,2,3,4 labels
    # mus_lst = []
    # for i in range(1, 5):
    #     # each label 1 time
    #     for _ in range(1):
    #         new_music = gan.generate(i)
    #         mus = new_music.astype(int)
    #         mus_lst.append(str(mus))
    #         # mid_new = arry2mid(mus)
    #         # mid_new.save('mid_label{}_number{}.mid'.format(i, _+1))
    # file_path = 'music.json'
    # with open(file_path, 'w') as file:
    #     json.dump(str(mus_lst), file)


if __name__ == "__main__":
    main()
