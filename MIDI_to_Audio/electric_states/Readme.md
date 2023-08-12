## 1. set states for a timber

e.g. python3 set_state.py martin_pick_nonoise_mono_body

(first string before _ should be in this dict:
GUITAR2ABBR = {'lespaul': 'LP', 'peregrine': 'PF', 'stratocaster': 'SC', 'semihollow': 'SH',
               'telecaster': 'TC', 'vintage': 'VC', 'eclipse': 'E',
               'luthier': 'L', 'gibson': 'SJ', 'taylor': 'T', 'martin': 'M'}
)


## 2. render acoustic or electric:

2.1 set indir and outdir in render_acoustic.py or render_electric_clean.py

2.2 follows run_acoustic.sh or run_electric_clean.sh


## 3. after rendering

python3 get_allfpath.py {outdir} 

to gen all the path for generated files ( one txt for audio, another txt for f0 )
