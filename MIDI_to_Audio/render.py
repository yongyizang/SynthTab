from re import X
import dawdreamer as daw
import numpy as np
import os,sys,glob,shutil
import pretty_midi
import multiprocessing
from tqdm import tqdm
import soundfile as sf
SAMPLE_RATE = 44100

cpu_cores = 8

# get the first argument
state = sys.argv[1]

BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
SYNTH_PLUGIN = '/Library/Audio/Plug-Ins/VST/AGL.vst'

tag = "luthier_" + state

path_indir = "/Users/colinzang/v1_val_train"
path_outdir = "/Volumes/ext/train_output"

# get all subfolders
subfolders = next(os.walk(path_indir))[1]
num_subfolders = len(subfolders)
print('num subfolders:', num_subfolders)

num_workers = cpu_cores
print(num_workers)
# split the subfolders into 6 parts
subfolders = np.array_split(subfolders, num_workers)
strings = 6

def render_audio(subfolders, state):
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth_name = "my_synth" + str(np.random.randint(100000))
    synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
    assert synth.get_name() == synth_name
    synth.load_state(state)

    for song in tqdm(subfolders, desc="Rendering"):
        # if the output file already exists, skip
        if os.path.exists(path_outdir + "/" + song.split("/")[-1].strip() + "/" + tag + ".flac"):
            continue
        
        try:
            song = path_indir + "/" + song
            if not os.path.exists(path_outdir + "/" + song.split("/")[-1].strip()):
                os.makedirs(path_outdir + "/" + song.split("/")[-1].strip())
            with open(song + "/tempo.txt", "r") as f:
                tempo = int(f.read().strip())
            song = song.split("/")[-1].strip()

            audios = []
            max_length = 0

            for string in range(1, strings + 1):
                if os.path.exists(path_indir + "/" + song + "/string_" + str(string) + ".mid"):
                    synth.load_midi(path_indir + "/" + song + "/string_" + str(string) + ".mid")
                    midi_data = pretty_midi.PrettyMIDI(path_indir + "/" + song + "/string_" + str(string) + ".mid")
                    duration = int(midi_data.get_end_time() + 5)
                    if duration > max_length:
                        max_length = duration
                    graph = [
                        (synth, [])
                    ]
                    engine.load_graph(graph)
                    engine.render(duration)
                    audios.append(engine.get_audio())

            # mix the audios
            audio_mix = np.zeros((2, int(max_length * SAMPLE_RATE)))
            for i in range(strings):
                audio_mix[:, :audios[i].shape[1]] += audios[i]
            audio_mix = audio_mix / strings
            if np.max(np.abs(audio_mix)) > 0.:
                audio_mix = audio_mix * 0.99 / np.max(np.abs(audio_mix))
            # save the audio
            # wavfile.write(path_outdir + "/" + song + "/" + tag + ".wav", SAMPLE_RATE, audio_mix.T)
            # as flac
            sf.write(path_outdir + "/" + song + "/" + tag + ".flac", audio_mix.T, SAMPLE_RATE, subtype='PCM_24')
        except Exception as e:
            with open("error.txt", "w") as f:
                f.write(song + " " + str(e) + "\n")
            continue

def main():
    multiprocessing.set_start_method('spawn')
    number_of_workers = cpu_cores
    processes = []
    for i in range(number_of_workers):
        p = multiprocessing.Process(target=render_audio, args=(subfolders[i], state))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
    
    print("Done")

if __name__ == '__main__':
    main()