from gan import MusicGAN
import torch
from midi_arr import *


# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.empty_cache()
#     print(torch.cuda.get_device_name())
#     music_data, label_data, align = write_music_data()
#     # print("midi done")
#     # gan = MusicGAN(align, device=device, epochs=10, batch_size=64)
#     # gan.train(music_data, label_data)
#     # # generate in 1,2,3,4 labels


def main():
    # TODO call save_data() once and comment it
    # save_data()
    music = load_music()
    label = load_label()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(torch.cuda.get_device_name())
    gan = MusicGAN(batch_size=256, epochs=200)
    gan.train(music, label)
    for i in range(1, 5):
        # each label 1 time
        for _ in range(2):
            new_music = gan.generate(i)
            # print(new_music)
            mus_name = "midi_label{}_number{}.mid".format(i, _+1)
            get_midi(new_music, mus_name)


if __name__ == "__main__":
    main()
