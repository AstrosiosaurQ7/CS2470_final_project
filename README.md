# CS2470 Final Project: Text to Music
### *text sentiment analysis and emotion-based music generation*

This is the Github repo for the project. For details, visit [DevPost](https://devpost.com/software/csci-2470-final-project-text-to-music)

## Team Member
Xiaoyan Liu, Yuhong Zhang, Ziyu Wang


## Data
The **EMOPIA** dataset is a specially curated collection designed for tasks related to music emotion recognition, with a particular focus on piano performances. The dataset encompasses key features as follows:

1) **Emotion Labels**: It classifies music pieces according to their emotional content, often employing models such as the valence-arousal framework. This framework interprets emotions along two dimensions: valence, ranging from positive to negative feelings, and arousal, varying from calm to excited states.

2) **Piano Music in MIDI Format**: The dataset comprises MIDI (Musical Instrument Digital Interface) files. MIDI serves as a standard protocol for recording and playing back musical performances, encapsulating details like note sequences, timing, and dynamics in a compact, precise manner. This attribute makes MIDI exceptionally conducive to computational analysis and synthesis, as it encodes musical instructions rather than audio waveforms. Despite this, waveforms in both the time and frequency domains remain crucial modalities explored in the feature space.

The dataset contains a total of 1,079 piano pieces, distributed across four emotional quadrants as follows: 250, 265, 253, and 310 pieces, respectively. The lengths of these music pieces range from 31.9 to 40.6 seconds, offering a broad spectrum for analysis in our final project.

**Data is not uploaded due to file size limitation**

## Model

#### Gan

1. Update a rough structure based on array-typed data, Pytorch.
2. Train on small sample of data(under construction)