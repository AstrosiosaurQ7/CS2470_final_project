import numpy as np
from new_gan import MusicCGAN
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
    save_data()

    music = load_music("music_array.npz")
    label = load_label("label_array.npz")

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.cuda.empty_cache()
    # print(torch.cuda.get_device_name())
    # # print("midi done")
    # gan = MusicGAN(batch_size=256, epochs=30, align=align, device=device)
    # gan.train(music_data, label_data)
    # for i in range(1, 5):
    #     # each label 1 time
    #     for _ in range(1):
    #         new_music = gan.generate(i)
    #         mus = new_music.astype(int)
    #         # mus_lst.append(str(mus))
    #         mid_new = arry2mid(mus)
    #         mid_new.save('mid_label{}_number{}.mid'.format(i, _+1))

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.empty_cache()
#     print(torch.cuda.get_device_name())
#     music_data, label_data, align = write_music_data()
#     gan = MusicGAN(batch_size=64, epochs=1, align=align, device=device)
#     new_music = gan.generate(1)
#     mus = new_music.astype(int)
#     mid_new = arry2mid(mus)
#     mid_new.save('mid_label{}_number{}.mid'.format(1, 1))


if __name__ == "__main__":
    main()
