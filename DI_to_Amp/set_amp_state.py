import dawdreamer as daw
import os,sys,glob,shutil
import numpy as np
import soundfile as sf
import librosa, json
import multiprocessing as mp

SAMPLE_RATE = 44100
BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.

def load_audio_file(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    audio = np.expand_dims(audio, axis=0)
    return audio

def get_audio_length(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    assert sr == SAMPLE_RATE
    return float(int(len(audio) / SAMPLE_RATE) + 1)

SYNTH_PLUGIN = '/Library/Audio/Plug-Ins/VST/BC Free Amp VST.vst'
engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
synth_name = "my_synth"
synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
assert synth.can_set_bus(2, 2)
synth.set_bus(2, 2)

graph = [
    (synth, ["DI"])
]

def run_audio(audio_path, input_dir, output_dir):
    DI = engine.make_playback_processor("DI", load_audio_file(wav))
    for item in range(8):
        synth.set_parameter(item+1, np.random.uniform(0,1))

    engine.load_graph(graph)
    engine.render(get_audio_length(audio))
    audio = engine.get_audio()
    audio = audio / np.max(np.abs(audio))

    state_dictionary = {}
    for i in range(8):
        state_dictionary[synth.get_parameter_name(i+1)] = synth.get_parameter(i+1)

    wav = wav.replace(input_dir, output_dir)
    if not os.path.exists(os.path.dirname(wav)):
        os.makedirs(os.path.dirname(wav))

    sf.write(wav, audio.T, SAMPLE_RATE)
    with open(wav[:-5] + "_amp_states.json", "w") as f:
        json.dump(state_dictionary, f)

if __name__ == "__main__":
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    zip_list = glob.glob(input_dir + "/*.zip")
    print("Found {} zip files.".format(len(zip_list)))
    for zip_file in zip_list:
        # Unzip
        print("Unzipping {}...".format(zip_file))
        os.system("unzip {} -d {}".format(zip_file, input_dir))
        # get zip directory
        zip_dir = zip_file[:-4]
        audio_list = glob.glob(zip_dir + "/**/*.flac", recursive=True)
        print("Found {} audio files.".format(len(audios)))
        print("Processing {} audio files...".format(len(audio_list)))
        with mp.Pool(mp.cpu_count()) as pool:
            tqdm(pool.imap_unordered(run_audio, audio_list, input_dir, output_dir), total=len(audio_list), desc="Processing audio files")
        # Remove zip directory
        shutil.rmtree(zip_dir)