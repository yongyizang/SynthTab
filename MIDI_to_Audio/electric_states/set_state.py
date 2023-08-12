import dawdreamer as daw
import os,sys,glob,shutil

SAMPLE_RATE = 44100
BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
output_name = sys.argv[1]

GUITAR2ABBR = {'lespaul': 'LP', 'peregrine': 'PF', 'stratocaster': 'SC', 'semihollow': 'SH',
               'telecaster': 'TC', 'vintage': 'VC', 'eclipse': 'E',
               'luthier': 'L', 'gibson': 'SJ', 'taylor': 'T', 'martin': 'M'}

GUITAR = output_name.split('/')[-1].split('_')[0]

SYNTH_PLUGIN = '/Library/Audio/Plug-Ins/VST/AG' + GUITAR2ABBR[GUITAR] + '.vst'
engine = daw.RenderEngine(SAMPLE_RATE, BUFFER_SIZE)
synth_name = "my_synth"
synth = engine.make_plugin_processor(synth_name, SYNTH_PLUGIN)
if len(sys.argv) == 3:
    synth.load_state(sys.argv[2])

synth.open_editor()

synth.save_state(output_name)
