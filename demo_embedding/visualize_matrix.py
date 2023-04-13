# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitar_transcription_inhibition.inhibition import load_inhibition_matrix, \
                                                       plot_inhibition_matrix, \
                                                       trim_inhibition_matrix

import amt_tools.tools as tools

# Regular imports
from matplotlib import rcParams

import matplotlib.pyplot as plt
import os


# Construct a path for loading the inhibition matrix
save_path = os.path.join('.', 'generated', 'matrices', 'synthtab_train_p128.npz')
# Load the inhibition matrix
matrix = load_inhibition_matrix(save_path)

# Initialize the default guitar profile
profile = tools.GuitarProfile(num_frets=19)

# Determine the number of strings and fret classes
num_strings = profile.get_num_dofs()
num_pitches = profile.num_pitches

# Remove extra frets from the inhibition matrix
matrix = trim_inhibition_matrix(matrix, num_strings, num_pitches, True)

# Change the font and font size for the plot
rcParams['font.family'] = 'monospace'
rcParams['font.size'] = 20

# Initialize a new figure with a large square shape
fig = plt.figure(figsize=(10, 10), tight_layout=True)

# Plot the inhibition matrix
fig = plot_inhibition_matrix(matrix, fig=fig)

# Call this to pause execution
plt.show(block=True)