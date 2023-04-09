# JAMS to MIDI module

This module provides a function to convert a JAMS file to a MIDI file. Typical commerical virtual instruments would utilize two types of messages for more dynamic performance: keyswitches to switch articulations, and also CC (Control Change) messages.

In our example code, we use modulation and pitch control messages to draw a modulation / pitch control curve. We also use keyswitches to switch articulations. Keyswitches' midi note number should be defined at `keyswitch_config.json` before running the code, since each target instrument may have different keys for different articulations. In our dataset, Ample Guitar M, Ample Guitar L, Ample Guitar T and Ample Guitar SJ all have the same keys for different articulations. So we use the same `keyswitch_config.json` for all of them.

For more detailed change to the existing code, check `generate_modulation()` and `generate_keyswitches()` in `JAMS-to-MIDI.py`. Right now, the code is running under the assumption that all input is a JAMS file, and it will output a midi file for each string that contains note information. Due to the limitation of Mido, which at the time of writing, only supports delta time, the program would generate a note track, a keyswitch track, a modulation/pitchbend track and an additional track (for overlapping notes with legato slides, required by Ample series), and merge them together in the end.

## Usage

```bash
python3 JAMS-to-MIDI.py <targetDir> <outputDir> <keyswitch_config>
```

Any error during generation will be captured in `error.txt` in the output directory.