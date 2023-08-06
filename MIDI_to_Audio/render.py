from re import X
import dawdreamer as daw
import numpy
import numpy as np
import os,sys,glob,shutil
import pretty_midi
import uuid
import multiprocessing
from tqdm import tqdm
import soundfile as sf
import librosa

SAMPLE_RATE = 44100

BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
GUITAR = "luthier"
SYNTH_PLUGIN = '/Library/Audio/Plug-Ins/VST/AGL.vst'

STRINGS = 6


def render_audio(subfolders, state, path_indir, path_outdir, extract_pitch=False):

    tag = "luthier_" + state

    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth_name = "my_synth" + str(uuid.uuid4())
    synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
    assert synth.get_name() == synth_name
    synth.load_state(state)

    for song in tqdm(subfolders, desc="Rendering"):
        # if the output file already exists, skip
        outdir = os.path.join(path_outdir, song.split("/")[-1].strip(), GUITAR)
        outsong = os.path.join(outdir, tag + ".flac")
        if os.path.exists(outsong):
            continue
        
        try:
            song = path_indir + "/" + song
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            with open(song + "/tempo.txt", "r") as f:
                tempo = int(f.read().strip())
            song = song.split("/")[-1].strip()

            audios = []
            max_length = 0

            for string in range(1, STRINGS + 1):
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
                    audio = engine.get_audio()

                    if extract_pitch:
                        f0, voiced_flag, voiced_probs, = librosa.pyin(audio.T[:, 0], fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=SAMPLE_RATE)
                        out_pitchf = os.path.join(outdir, tag + "_string_" + str(string))
                        numpy.save(out_pitchf, f0)
                    audios.append(audio)

            # mix the audios
            audio_mix = np.zeros((2, int(max_length * SAMPLE_RATE)))
            for i in range(len(audios)):
                audio_mix[:, :audios[i].shape[1]] += audios[i]
            audio_mix = audio_mix / len(audios)
            if np.max(np.abs(audio_mix)) > 0.:
                audio_mix = audio_mix * 0.99 / np.max(np.abs(audio_mix))
            # save the audio
            # wavfile.write(path_outdir + "/" + song + "/" + tag + ".wav", SAMPLE_RATE, audio_mix.T)
            # as flac
            sf.write(outsong, audio_mix.T, SAMPLE_RATE, subtype='PCM_24')
        except Exception as e:
            with open("error.txt", "w") as f:
                f.write(song + " " + str(e) + "\n")
            continue


if __name__ == '__main__':
    cpu_cores = 6

    # get the first argument
    state = sys.argv[1]

    path_indir = "../JAMS_to_MIDI/allMIDIs"
    path_outdir = "./test"
    # get all subfolders
    subfolders = next(os.walk(path_indir))[1]

    acoustic_folders = []
    # typeset = set()
    for folder in subfolders:
        assert len(folder.split('__')) == 3
        instrum = folder.split('__')[1]
        splits = instrum.split(' ')
        # typeset.add(' '.join(splits[2:]))

        if splits[-1] != 'Guitar':
            continue
        if splits[2] == 'Acoustic':
            acoustic_folders.append(folder)

    num_subfolders = len(subfolders)
    print('num subfolders:', num_subfolders)
    num_acoustic_folders = len(acoustic_folders)
    print('num acoustic_folders:', num_acoustic_folders)


    num_workers = cpu_cores
    print(num_workers)
    # split the subfolders into n_worker parts
    subsubfolders = np.array_split(acoustic_folders, num_workers)

    multiprocessing.set_start_method('spawn')
    number_of_workers = cpu_cores
    processes = []
    for i in range(number_of_workers):
        p = multiprocessing.Process(target=render_audio, args=(subsubfolders[i], state, path_indir, path_outdir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done")
