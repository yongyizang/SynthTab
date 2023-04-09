# Demo Data (for developing) of SynthTab
## File format

SynthTab is released as both `train` and `validation` data. Training portion contains multiple MP3 format files with multiple microphone signals, while validation portion contains flac files. The file format is as follows:

```
├── train
│   ├── [song name]
│   │   ├── ground_truth.jams (This is the ground truth tablature)
│   │   ├── [guitar name] (There may be multiple guitars)
│   │   │   ├── [guitar mic signal] (There are multiple microphone signals)

```

Validation portion is similar to training portion, except that it contains flac files instead of mp3 files.

## Augmentation
See augmentation code in `augment.py`, which requires `Pedalboard` to run as its dependency.