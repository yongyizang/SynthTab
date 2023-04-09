import dawdreamer as daw
import os,sys,glob,shutil
SAMPLE_RATE = 44100
BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
output_name = sys.argv[1]
load_state_name = sys.argv[2]
SYNTH_PLUGIN = '/Library/Audio/Plug-Ins/VST/AGL.vst'
engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
synth_name = "my_synth"
synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
synth.open_editor()
synth.save_state(sys.argv[1])