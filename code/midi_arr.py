import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage
# from mido import MIDITime
import os
import string
import numpy as np
import pandas as pd


def get_song_msg(mid):
    one_song_msg = []
    for msg in mid.tracks[1]:
        if msg.type == 'note_on':
            note = msg.note
            velocity = msg.velocity
            time = msg.time
            temp = [note,velocity,time]
            one_song_msg.append(temp)
    return one_song_msg


def get_emo(label, target):
    for i in range(len(label)):
        if label.iloc[i,0] == target:
            return label.iloc[i,1]


def get_label(label_path):
    label = pd.read_csv(label_path, delimiter='\t')
    # seperate the first column
    label[['name', '4Q', 'annotator']] = label['ID,4Q,annotator'].str.split(',', expand=True)
    # drop the original column
    label.drop(columns=['ID,4Q,annotator'], inplace=True) 
    return label


def get_music_data(folder_path, label_path):
    music_data = []
    label_data = []
    align_length = 500
    for file_name in os.listdir(folder_path):
        # Construct the full path to the MIDI file
        midi_file_path = os.path.join(folder_path, file_name)
        # Load the MIDI file
        mid = MidiFile(midi_file_path)
        song_msg = get_song_msg(mid)

        #  align the data 500
        if len(song_msg) > align_length:
            result_array = song_msg[:align_length]
        else:
            num_to_pad = align_length - len(song_msg)
            result_array = np.pad(song_msg, ((0, num_to_pad), (0, 0)), mode='constant')

        music_data.append(np.array(result_array))
        # label
        filename_without_extension = os.path.splitext(file_name)[0]
        labels = get_label(label_path)
        # print(labels)
        emo = get_emo(labels, filename_without_extension)
        label_data.append(emo)

    return music_data, np.array(label_data)


def save_data():
    # TODO CHANGE PATH HERE
    label_general_path = '../data/EMOPIA/label.csv'
    folder_general_path = '../data/EMOPIA/midis'
    music_data, label_data = get_music_data(folder_general_path, label_general_path)
    np.savez('music_array.npz', *music_data)
    np.savez('label_array.npz', label_data)
    return


def load_music(path):
    data = np.load(path)
    musics = [data[key] for key in data]
    return musics


def load_label(path):
    data = np.load(path)
    labels = data['number_array']
    return labels


def get_midi(events, output_file):
    mid = mido.MidiFile()
    track = mido.MidiTrack()

    # Set the tempo to default 120 BPM
    track.append(mido.MetaMessage('set_tempo', tempo=500000))

    # Iterate through the events and convert them to MIDI messages
    for event in events:
        note, velocity, time = event
        # Create a note_on message
        note_on = mido.Message('note_on', note=note, velocity=velocity, time=time)
        track.append(note_on)

    # Add the track to the MIDI file
    mid.tracks.append(track)

    # Save the MIDI file
    mid.save(output_file)

# MIDI events / list format
# events = music[1]

# Output MIDI file name
# output_file = "output.mid"

# Reconstruct MIDI and save to file
# get_midi(events, output_file)
