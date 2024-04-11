# CS2470_final_project
text sentiment analysis and emotion-based music generation

## Data
The **EMOPIA** dataset is a specially curated collection designed for tasks related to music emotion recognition, with a particular focus on piano performances. The dataset encompasses key features as follows:

1) **Emotion Labels**: It classifies music pieces according to their emotional content, often employing models such as the valence-arousal framework. This framework interprets emotions along two dimensions: valence, ranging from positive to negative feelings, and arousal, varying from calm to excited states.

2) **Piano Music in MIDI Format**: The dataset comprises MIDI (Musical Instrument Digital Interface) files. MIDI serves as a standard protocol for recording and playing back musical performances, encapsulating details like note sequences, timing, and dynamics in a compact, precise manner. This attribute makes MIDI exceptionally conducive to computational analysis and synthesis, as it encodes musical instructions rather than audio waveforms. Despite this, waveforms in both the time and frequency domains remain crucial modalities explored in the feature space.

The dataset contains a total of 1,079 piano pieces, distributed across four emotional quadrants as follows: 250, 265, 253, and 310 pieces, respectively. The lengths of these music pieces range from 31.9 to 40.6 seconds, offering a broad spectrum for analysis in our final project.

## Methodology
Text Sentiment Analysis: We intend to leverage a Large Language Model, specifically by employing the ChatGpt API, to conduct a comprehensive analysis of sentiment labels derived from a text corpus. Our approach involves meticulously crafting prompts that enable the precise categorization of sentiments into one of four distinct quadrants. This method not only aims to utilize the advanced capabilities of Gpt for sentiment analysis but also to refine the process through strategic prompt engineering. This will ensure a nuanced understanding of sentiment that aligns with our predefined emotional quadrants, facilitating a more targeted and effective analysis.

Emotion-based Generation: To generate melodies based on emotional expression with audibility, we aim to train models using the EMOPIA dataset, which contains melodies paired with emotion labels. Our current dataset is in MIDI format, and we intend to preprocess it using mido. Our strategy involves selecting high-performing models from RNNs, Transformers, and GANs. These models will then be used to generate melodies that reflect emotional while ensuring audible quality. Since all of us are new to generative models, we are open to any new methods which can improve and optimize our process.
