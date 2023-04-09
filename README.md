# SynthTab: A Synthesized Dataset for Improved Guitar Tablature Transcription

Official GitHub Repository for (insert paper link here).

## Structure
This repository is modular, as every part of it can be re-used to generate other similar dataset using our methodology. The repository is structured as follows:

`gp_to_JAMS` folder contains all necessary code to generate JAMS files from Guitar Pro files.

`JAMS_to_MIDI` folder contains all necessary code to generate MIDI files (per-string) from JAMS files.

`MIDI_to_audio` folder contains all necessary code to generate audio files (per-guitar-mic) from MIDI files - they are currently designed to take the output from `JAMS_to_MIDI` folder, but can be easily further customized.

`demo_data` contains a small portion of our data that allows for development of the algorithm; once your algoirthm is ready, simply download your dataset and follow the similar file structure.

`demo_embedding` is where we put our benchmark pre-trained models, with a simple demo script for training and running evaluation.

In each folder you will find the corresponding README file, explaining how the content of that folder works.