from gan import MusicGAN
import torch
from midi_array import *
import gc


def write_music_data():
    label_general_path = '../data/EMOPIA/label.csv'
    folder_general_path = '../data/EMOPIA/midis'

    return get_music_data(folder_general_path, label_general_path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    gc.collect()
    print(torch.cuda.get_device_name())
    music_data, label_data, align = write_music_data()
    print("midi done")
    gan = MusicGAN(align, device=device, epochs=10, batch_size=64)
    gan.train(music_data, label_data)
    # generate in 1,2,3,4 labels
    for i in range(1, 5):
        # each label 3 times
        for _ in range(3):
            new_music = gan.generate(i)
            mid_new = arry2mid(new_music)
            mid_new.save('mid_label{}_number{}.mid'.format(i, _+1))


if __name__ == "__main__":
    main()
