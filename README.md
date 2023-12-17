# SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription
Yongyi Zang*, Yi Zhong* (Equal Contribution), Frank Cwitkowitz, Zhiyao Duan

We created a large-scale synthesized guitar tablature dataset to address the low-resource problem in guitar tablature transcription. This repository contains code for our rendering pipeline, along with our baseline models (TabCNN, TabCNN+) and our trained embeddings.

[[Project Website](https://synthtab.dev/)] [[Paper Link](https://arxiv.org/pdf/2309.09085.pdf)] 

## Updates
- **Dec 2023**: SynthTab is accepted at ICASSP 2024!

## Cite Us
If you use SynthTab as part of your research, please cite us according to the following BibTeX:
```
@inproceedings{synthtab2024,
  title={SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription},
  author={Zang, Yongyi and Zhong, Yi and Cwitkowitz, Frank and Duan, Zhiyao}
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

## Downloading SynthTab

The development set of SynthTab is available at [here](https://rochester.app.box.com/v/SynthTab-Dev).

Due to the large size of SynthTab, we temporarily host it at this [MEGA folder](https://mega.nz/folder/ZG9BgB6a#BnZ5MruFQsRdgraOBYh60Q). Additionally, we provide a [Baidu Netdisk link](https://pan.baidu.com/s/1PF8EAHkHmFhx7ySVRbWMDA) (Password: gjwq) for easy access of the same content for several regions. Total file size is close to and less than 2 TB.

SynthTab is released with CC BY-NC 4.0 license (learn more about it [here](https://creativecommons.org/licenses/by-nc/4.0/deed.en)).

File structure is as follows:
```
SynthTab
|---all_jams_midi_V2_60000_tracks.zip
|---acoustic
|---electric_clean
|---electric_distortion
|---electric_muted
```

The JAMS files are stored separatedly in `all_jams_midi_V2_60000_tracks.zip`. It is relatively small at around 1 GB. `acoustic`, `electric_clean`, `electric_distortion` and `electric_muted` all contains different timbres as `*.zip` files inside them.

Within each song's rendered files, we also provide per-string extracted fundamental frequency (stored as `*.pkl` files). We used the YIN algorithm for this. See `MIDI_to_Audio/render.py` for the exact implementation of the extraction process.

## Structure
This repository is modular, as every part of it can be re-used to generate other similar dataset using our methodology. The repository is structured as follows:

`gp_to_JAMS` folder contains all necessary code to generate JAMS files from Guitar Pro files.

`JAMS_to_MIDI` folder contains all necessary code to generate MIDI files (per-string) from JAMS files.

`MIDI_to_audio` folder contains all necessary code to generate audio files (per-guitar-mic) from MIDI files - they are currently designed to take the output from `JAMS_to_MIDI` folder, but can be easily further customized.

`demo_data` contains a small portion of our data that allows for development of the algorithm; once your algoirthm is ready, simply download your dataset and follow the similar file structure.

`demo_embedding` is where we put our benchmark pre-trained models, with a simple demo script for training and running evaluation.

In each folder you will find the corresponding README file, explaining how the content of that folder works.