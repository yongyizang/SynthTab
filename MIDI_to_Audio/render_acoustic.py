import pdb
import random
import pickle
import dawdreamer as daw
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
STATE = sys.argv[1]
GUITAR = STATE.split('/')[-1].split('_')[0]
GUITAR_2 = '_'.join(STATE.split('/')[-1].split('_')[:2])

GUITAR2ABBR = {'luthier': 'L', 'gibson': 'SJ', 'taylor': 'T', 'martin': 'M'}
SYNTH_PLUGIN = \
    '/Library/Audio/Plug-Ins/VST/AG' + GUITAR2ABBR[GUITAR] + '.vst'

STRINGS = 6


def render_audio(subfolders, state, path_indir, path_outdir, extract_pitch=True, mono=True):

    tag = state.split('/')[-1]
    engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
    synth_name = "my_synth" + str(uuid.uuid4())
    synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
    assert synth.get_name() == synth_name
    synth.load_state(state)

    for song in tqdm(subfolders, desc="Rendering"):
        # if the output file already exists, skip
        outdir = os.path.join(path_outdir, GUITAR_2, song.split("/")[-1].strip())
        outsong = os.path.join(outdir, tag + ".flac")
        if os.path.exists(outsong):
            continue
        
        try:
            song = path_indir + "/" + song
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            # with open(song + "/tempo.txt", "r") as f:
            #     tempo = int(f.read().strip())
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
                        # Iterate over notes in the instrument track
                        notes_f0 = []
                        for note in midi_data.instruments[0].notes: # skip the first inserted pseudo note?
                            start = note.start
                            end = note.end
                            duration = end - start
                            start_index = round(note.start * SAMPLE_RATE)
                            end_index = round(note.end * SAMPLE_RATE)
                            f0 = librosa.yin(audio[0, start_index:end_index],
                                                      fmin=librosa.note_to_hz('C2'),
                                                      fmax=librosa.note_to_hz('C7'),
                                                      sr=SAMPLE_RATE,
                                                      )
                            f0[f0 == 2100.] = 0.0
                            f0[np.isnan(f0)] = 0.0
                            notes_f0.append(f0)

                            #print(f"Note {note.pitch} has start：{start}, end :{end} duration: {duration} seconds")
                            #print(f"Note {f0}")

                        # Here we'll use a 5 millisecond hop length
                        # hop_length = int(SAMPLE_RATE / 200.)
                        #
                        # # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
                        # # This would be a reasonable range for speech
                        # fmin = 65
                        # fmax = 2006
                        #
                        # # Select a model capacity--one of "tiny" or "full"
                        # model = 'tiny'
                        # # Compute pitch using first gpu
                        # f0 = torchcrepe.predict(torch.from_numpy(audio[0]).unsqueeze(0),
                        #                         SAMPLE_RATE,
                        #                         hop_length,
                        #                         fmin,
                        #                         fmax,
                        #                         model,
                        #                         batch_size=1,
                        #                         device='cpu')
                        # f0 = f0[0].numpy()

                        #if mono:
                        #    f0 = librosa.yin(audio[0],
                        #                     fmin=librosa.note_to_hz('C2'),
                        #                     fmax=librosa.note_to_hz('C7'),
                        #                     sr=SAMPLE_RATE)
                        #else:
                        #    f0 = librosa.yin(audio,
                        #                     fmin=librosa.note_to_hz('C2'),
                        #                     fmax=librosa.note_to_hz('C7'),
                        #                     sr=SAMPLE_RATE)
                            #
                        # non voice will be fmax for yin, update to 0
                        out_pitchf = os.path.join(outdir, tag + "_string_" + str(string) + "_pitch.pkl")
                        with open(out_pitchf, 'wb') as f:
                            pickle.dump(notes_f0, f)

                        #pdb.set_trace()
                        #np.save(out_pitchf, f0)

                    audios.append(audio)

            # mix the audios
            audio_mix = np.zeros((2, int(max_length * SAMPLE_RATE)))
            for i in range(len(audios)):
                audio_mix[:, :audios[i].shape[1]] += audios[i]
            audio_mix = audio_mix / len(audios)

            if np.max(np.abs(audio_mix)) > 0.:
                audio_mix = audio_mix * 0.99 / np.max(np.abs(audio_mix))


            sf.write(outsong, audio_mix[0].T, SAMPLE_RATE, subtype='PCM_24')
        except Exception as e:
            with open("error.txt", "w") as f:
                f.write(song + " " + str(e) + "\n")
            continue


if __name__ == '__main__':

    # get the first argument

    path_indir = "../../outall/"
    #path_indir = "../JAMS_to_MIDI/outtest/"
    path_outdir = "../../syn_out/acoustic/"
    #path_outdir = "./test/"

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

    #render_audio(acoustic_folders, STATE, path_indir, path_outdir, extract_pitch=True, mono=True)
    #exit()


    num_workers = 20
    print('num_workers:', num_workers)
    # split the subfolders into n_worker parts
    random.shuffle(acoustic_folders)

    subsubfolders = np.array_split(acoustic_folders, num_workers)

    multiprocessing.set_start_method('spawn')
    processes = []
    for i in range(num_workers):
        p = multiprocessing.Process(target=render_audio, args=(subsubfolders[i], STATE, path_indir, path_outdir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Done")
