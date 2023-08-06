# -*- coding: utf-8 -*-

import json, os, sys, shutil, random

import numpy as np
from numpy import size
import mido as Mido
from tqdm import tqdm
import matplotlib.pyplot as plt

def jams_to_midi(jams_dir, midi_output_folder, keyswitch_config):
    # get notes data
    json_data = open(jams_dir).read()
    strings = json.loads(json_data)["annotations"]
    
    tempo_changes = []
    
    for string in strings:
        if string["namespace"] == "tempo":
            for tempo in string["data"]:
                tempo_changes.append((tempo["time"], tempo["value"]))
                
    for string in strings:
        if string["namespace"] == "tempo":
            continue
        # 1st string will be 1, etc.
        string_index = string["sandbox"]['string_index']
        # note the MIDI number of what the open string is tuned to. Used to normalize the MIDI numbers of the notes.
        open_tuning = string["sandbox"]['open_tuning']
        notes = string["data"]
        if (len(notes) == 0):
            continue
        
        # Start by creating a MIDI file with a single track.
        midi_file = Mido.MidiFile()
        # Set to 96 ticks per beat.
        midi_file.ticks_per_beat = 960
        
        midi_track = Mido.MidiTrack()
        midi_track.append(Mido.MetaMessage('set_tempo', tempo=Mido.bpm2tempo(tempo_changes[0][1]), time=0))

        keyswitch_track = Mido.MidiTrack()
        keyswitch_track.append(Mido.MetaMessage('set_tempo', tempo=Mido.bpm2tempo(tempo_changes[0][1]), time=0))
        
        modulation_track = Mido.MidiTrack()
        modulation_track.append(Mido.MetaMessage('set_tempo', tempo=Mido.bpm2tempo(tempo_changes[0][1]), time=0))

        pitch_bend_track = Mido.MidiTrack()
        pitch_bend_track.append(Mido.MetaMessage('set_tempo', tempo=Mido.bpm2tempo(tempo_changes[0][1]), time=0))
        
        
        # To be able to sync everything, create a blank note at the beginning of the track.
        midi_track.append(Mido.Message('note_on', note=keyswitch_config["sustain"], velocity=100, time=0, channel=string_index-1))
        midi_track.append(Mido.Message('note_off', note=keyswitch_config["sustain"], velocity=100, time=2, channel=string_index-1))
        
        organized_notes = []
        for note in notes:
            noteValue = note["value"]["fret"] + open_tuning
            velocity = note["value"]["velocity"]
            time = int(note["time"])
            duration = int(note["duration"])
            effect = parse_note_effect(note["value"])
            noteObj = {
                "note": noteValue,
                "velocity": velocity,
                "time": time,
                "duration": duration,
                "effect": effect,
                "raw_effect": note["value"]
            }
            organized_notes.append(noteObj)
            
        # sort by time
        organized_notes = sorted(organized_notes, key=lambda k: k['time'])
        
        # generate keyswitches and modulation based on the organized notes
        keyswitches, notes_copy, modified_notes = generate_keyswitches(organized_notes, keyswitch_config, delay=10)
        curve_sample_interval = 100
        modulation_curve, pitch_curve = generate_modulation(organized_notes, curve_sample_interval=curve_sample_interval)
        
        # everything is ready, now we can start writing to the MIDI file
        # due to "delta-only" write mode for Mido, we write every articulation in a separate track, and then merge them together.
        current_tick = 0
        for keyswitch in keyswitches:
            keyswitch_track.append(Mido.Message('note_on', note=keyswitch["value"], velocity=127, time=keyswitch["time"] - current_tick, channel=string_index-1))
            keyswitch_track.append(Mido.Message('note_off', note=keyswitch["value"], velocity=127, time=2, channel=string_index-1))
            current_tick = keyswitch["time"] + 2
        
        current_tick = 0
        for i in range(len(modulation_curve)):
            modulation_track.append(Mido.Message('control_change', control=1, value=int(modulation_curve[i]), time=i*curve_sample_interval - current_tick, channel=string_index-1))
            current_tick = i*curve_sample_interval
            
        current_tick = 0
        for event in pitch_curve:
            pitch_bend_track.append(Mido.Message('pitchwheel', pitch=int(event["pitch"]), time=int(event["time"]) - current_tick, channel=string_index-1))
            current_tick = int(event["time"])
        
        # finally we are ready to write the notes to the MIDI file
        current_tick = 0
        for note in notes_copy:
            midi_track.append(Mido.Message('note_on', note=note["note"], velocity=note["velocity"], time=note["time"] - current_tick, channel=string_index-1))
            midi_track.append(Mido.Message('note_off', note=note["note"], velocity=note["velocity"], time=note["duration"], channel=string_index-1))
            current_tick = note["time"] + note["duration"]
        
        additional_track = Mido.MidiTrack()
        additional_track.append(Mido.MetaMessage('set_tempo', tempo=Mido.bpm2tempo(tempo_changes[0][1]), time=0))
        current_tick = 0
        for note in modified_notes:
            additional_track.append(Mido.Message('note_on', note=note["note"], velocity=note["velocity"], time=note["time"] - current_tick, channel=string_index-1))
            additional_track.append(Mido.Message('note_off', note=note["note"], velocity=note["velocity"], time=note["duration"], channel=string_index-1))
            current_tick = note["time"] + note["duration"]
        
        merged_track = Mido.merge_tracks([midi_track, keyswitch_track, modulation_track, pitch_bend_track, additional_track])
        midi_file.tracks.append(merged_track)
        midi_file.save(os.path.join(midi_output_folder, f'string_{string_index}.mid'))
        
    return tempo_changes[0][1]
        

def parse_note_effect(effect):
    # check if effect has key 'slide'
    if 'slide' in effect:
        if 'outDownwards' in effect['slide']:
            return 'outDownwards'
        elif 'intoFromBelow' in effect['slide']:
            return 'intoFromBelow'
        elif 'shiftSlideTo' in effect['slide'] or 'legatoSlideTo' in effect['slide']:
            return 'shiftSlideTo'
    elif 'palmMute' in effect and effect['palmMute'] == True:
        return 'palm_mute'
    elif 'harmonic' in effect:
        return 'harmonic'
    elif 'vibrato' in effect and effect['vibrato'] == True:
        return 'vibrato'
    elif 'bend' in effect:
        return 'bend'
    else:
        return 'sustain'
    
def generate_keyswitches(notes, keyswitch_config, delay=0):
    keyswitches = []
    modified_notes = []
    notes_copy = notes.copy()
    # at the beginning, we force it to reset to sustain.
    # at each note, if it is mute, then we add a keyswitch to mute.
    # if it is harmonic, then we add a keyswitch to harmonic.
    # however, if it is slide, we need keyswitch at its previous note.
    # therefore, we go through the notes, and add keyswitches to both the current note and the previous note. 
    for i in range(len(notes)-1, 1, -1):
        note = notes[i]
        prev_note = notes[i-1] if i > 0 else None
        # parse keyswitches at the current note
        if note["effect"] == 'harmonic':
            keyswitches.append({
            "time": note["time"] - delay,
            "value": keyswitch_config["harmonic"],
            })
        elif note["effect"] == 'palm_mute':
            keyswitches.append({
                "time": note["time"] - delay,
                "value": keyswitch_config["palm_mute"],
            })
        elif note["effect"] == 'intoFromBelow':
            keyswitches.append({
                "time": note["time"] - delay,
                "value": keyswitch_config["slide_in_out"],
            })
        elif note["effect"] == 'outDownwards':
            keyswitches.append({
                "time": note["time"] + note["duration"] - delay,
                "value": keyswitch_config["slide_in_out"],
            })
        # parse keyswitches at the previous note
        if prev_note and prev_note["note"] != note["note"]:
            if note["effect"] == 'shiftSlideTo':
                keyswitches.append({
                    "time": note["time"] - delay,
                    "value": keyswitch_config["legato_slide"],
                })
            # make sure two notes overlap
            notes[i-1]["duration"] += random.randint(5, 10)
            modified_notes.append(notes[i-1])
            # remove the previous note from notes
            notes_copy.pop(i-1)
    return keyswitches, notes_copy, modified_notes

def generate_modulation(notes, curve_sample_interval=100, max_bend=8, max_bend_value=8191):
    # we generate two modulation curves, one for vibrato (CC 1), and one for pitch bend.
    raw_vibrato_events = []
    raw_pitch_bend_events = []
    # pick out two types of notes: vibrato and bend.
    for i in range(len(notes)):
        note = notes[i]
        if note["effect"] == 'vibrato':
            # vibrato is a special case, because it's not a keyswitch, and need to draw CC modulation.
            # we draw a CC modulation from 0 to 127, and back to 0.
            # the duration of the CC modulation is the same as the note.
            raw_vibrato_events.append({
                "time": note["time"],
                "duration": note["duration"],
                "mode": "vibrato",
            })
        elif note["effect"] == "bend":
            bendcurve = []
            for i in range(size(note["raw_effect"]["bend"]["points"]["position"])):
                bendcurve.append([float(note["raw_effect"]["bend"]["points"]["position"][i]), float(note["raw_effect"]["bend"]["points"]["height"][i])])
            raw_pitch_bend_events.append({
                "time": note["time"],
                "duration": note["duration"],
                "mode": "bend",
                "bendcurve": bendcurve,
            })
            
    # with the raw events, we generate the actual modulation curves.
    # longest note duration
    if notes != []:
        total_duration = notes[-1]["time"] + notes[-1]["duration"]
        # here, grid_size denotes the number of points we want to sample from the curve, which is to make it more smooth. This is customizable.
        grid_size = curve_sample_interval
        # we generate a grid of time points, and sample the curve at those points.
        time_point_size = total_duration // grid_size
        modulation_curve = np.zeros((1, time_point_size))
        pitch_curve = np.zeros((1, time_point_size))
        
        # we start by processing modulation curve.
        
        # go through all modulation events, and insert the "vibrato" ones.
        for event in raw_vibrato_events:
            start_time = event["time"]
            end_time = event["time"] + event["duration"]
            start_index = start_time // time_point_size
            end_index = end_time // time_point_size
            random_perturbation = random.randint(-15, 0)
            modulation_curve[0, start_index:end_index] = 127 + random_perturbation
        
        # now, go through all the points in modulation curve that doesn't have a vibrato event, and fill in the gaps with small random perturbations. This helps to simulate real playing.
        for i in range(len(modulation_curve[0])):
            if modulation_curve[0, i] == 0:
                modulation_curve[0, i] = random.randint(0, 15)
                
        # with all these done, we use a running average of 10 points to smooth out the curve.
        new_modulation_curve = np.zeros((1, time_point_size))
        for i in range(5, len(modulation_curve[0]) - 5):
            new_modulation_curve[0, i] = np.mean(modulation_curve[0, i-5:i+5])
        modulation_curve = new_modulation_curve
        
        pitch_events = []
        
        for event in raw_pitch_bend_events:
            start_time = event["time"]
            end_time = event["time"] + event["duration"]
            bendcurve = event["bendcurve"]
            pitch_events.append({
                "time": start_time - 1,
                "pitch": 0,
            })
            for i in range(len(bendcurve)):
                event_time = start_time + bendcurve[i][0] * event["duration"]
                event_pitch = bendcurve[i][1] * max_bend_value / max_bend
                pitch_events.append({
                    "time": event_time,
                    "pitch": event_pitch,
                })
            pitch_events.append({
                "time": end_time + 1,
                "pitch": 0,
            })
            
        return modulation_curve[0], pitch_events
    else:
        return [], []

if __name__ == "__main__":
    targetDir = sys.argv[1]
    outputDir = sys.argv[2]
    with open(sys.argv[3], 'r') as f:
        keyswitch_config = json.load(f)
    
    # filter out if args are not 3
    if len(sys.argv) != 4:
        print("Usage: python3 JAMS-to-MIDI.py <targetDir> <outputDir> <keyswitch_config>")
        exit()
    
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    
    # for all the jams files in the target directory, convert them to midi files.
    allJams = []
    for root, dirs, files in os.walk(targetDir):
        for file in files:
            if file.endswith(".jams"):
                allJams.append(os.path.join(root, file))

    for jam in tqdm(allJams, desc="Converting JAMS to MIDI"):
        outdir_name = jam.split("/")[-2].replace('__', '_').strip('_') + '__' + jam.split("/")[-1].split(".jams")[0].strip().strip('_') + "__midi"
        output_dir = os.path.join(outputDir, outdir_name)
        # copy the jam file to the output directory.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copy(jam, output_dir)
        try:
            tempo = jams_to_midi(jam, output_dir, keyswitch_config)
            with open(os.path.join(output_dir, "tempo.txt"), "w") as f:
                f.write(str(tempo))
        except Exception as e:
            print("Error converting file: ", jam)
            with open(outputDir + "/error.txt", "a") as f:
                f.write(jam + " " + str(e) + "\n")
            print(e)
            continue
