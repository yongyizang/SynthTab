# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitarpro_utils import validate_gpro_track, \
                            parse_notes_gpro_track, \
                            VALID_INSTRUMENTS

# Regular imports
import guitarpro
import jams
import os


VALID_GP_EXTS = ['.gp3', '.gp4', '.gp5']
INVALID_EXTS = ['.pygp', '.gp2tokens2gp']


def clean_jams(base_dir):
    """
    Walk through a base directory and remove all JAMS files.

    Parameters
    ----------
    base_dir : string
      Path to the base directory to recursively search
    """

    # Traverse through all paths within the base directory
    for dir_path, dirs, files in os.walk(base_dir):
        # Loop through files in directory and remove any JAMS files
        for f in [f for f in files if os.path.splitext(f)[-1] == '.jams']:
            os.remove(os.path.join(dir_path, f))

        # Remove directories with no files or sub-directories
        if not len(os.listdir(dir_path)):
            os.rmdir(dir_path)


def get_valid_files(base_dir):
    """
    Walk through a base directory and keep track of all relevant GuitarPro files.

    Parameters
    ----------
    base_dir : string
      Path to the base directory to recursively search

    Returns
    ----------
    tracked_files : list of str
      List of file names found
    tracked_dirs : list of str
      List of directories containing tracked files
    """

    # Keep track of valid GuitarPro files
    tracked_files, tracked_dirs = list(), list()

    # Traverse through all paths within the base directory
    for dir_path, dirs, files in os.walk(base_dir):
        # Ignore directories with no files (only directories)
        if not len(files):
            continue

        # Obtain a list of valid GuitarPro files within the current directory
        valid_files = sorted([f for f in files
                              if os.path.splitext(f)[-1] in VALID_GP_EXTS
                              and INVALID_EXTS[0] not in f
                              and INVALID_EXTS[1] not in f])

        # Add valid files to tracked list
        tracked_files += valid_files

        # Update the tracked paths
        tracked_dirs += [dir_path] * len(valid_files)

    # Order files alphabetically w.r.t. path
    tracked_dirs, tracked_files = zip(*sorted(zip(tracked_dirs, tracked_files)))

    return tracked_files, tracked_dirs


def write_jams_guitarpro(gpro_path, jams_dir):
    """
    Convert a GuitarPro file to a JAMS file specifying notes for each track.

    Parameters
    ----------
    gpro_path : string
      Path to a preexisting GuitarPro file to convert
    jams_dir : bool
      Directory under which to place the JAMS files
    """

    # Extract the GuitarPro data from the file
    gpro_data = guitarpro.parse(gpro_path)

    # Loop through the instrument tracks in the GuitarPro data
    for t, gpro_track in enumerate(gpro_data.tracks):
        # Make sure the GuitarPro file can be processed for symbolic datasets
        if validate_gpro_track(gpro_track):
            # Make sure the JAMS directory exists
            os.makedirs(jams_dir, exist_ok=True)

            # Extract the notes and duration of the track, given a global tempo
            note_tracker, tempo_changes = parse_notes_gpro_track(gpro_track, gpro_data.tempo)

            # Write the note predictions to a JAMS object
            jam = note_tracker.write_jams()

            # Add the total duration to the top-level file meta-data
            jam.file_metadata.duration = tempo_changes[-1][0] - tempo_changes[0][0]

            # Write track meta-data to the JAMS object
            jam.sandbox.update(instrument=gpro_track.channel.instrument,
                               fret_count=gpro_track.fretCount)

            # Create a label for the track based on the encoded instrument
            track_label = f'{t + 1} - {VALID_INSTRUMENTS[gpro_track.channel.instrument]}'

            # Create a new annotation for all tempo changes
            all_tempo_data = jams.Annotation(namespace='tempo')

            # Loop through tempo changes
            for i in range(1, len(tempo_changes)):
                # Extract the tempo change information
                tick, tempo = tempo_changes[i - 1]
                # Compute the amount of ticks for which tempo is valid
                duration = tempo_changes[i][0] - tempo_changes[i - 1][0]
                # Add the tempo change to the top-level JAMS data
                all_tempo_data.append(time=tick, value=tempo, duration=duration, confidence=1.0)

                if len(tempo_changes) > 2:
                    # Slice the top-level jam to the duration of the tempo
                    jam_slice = jam.slice(tick, tick + duration)

                    # Create a new annotation for the tempo of the slice
                    tempo_data = jams.Annotation(namespace='tempo')
                    # Add an entry denoting the tempo to the JAMS slice
                    tempo_data.append(time=0, value=tempo, duration=duration, confidence=1.0)

                    # Add the annotation to the JAMS object
                    jam_slice.annotations.append(tempo_data)

                    # Create a new directory for JAMS annotations sliced by tempo
                    tempo_split_dir = os.path.join(jams_dir, f'{track_label} - tempo_slices')

                    # Make sure the directory for the slices exists
                    os.makedirs(tempo_split_dir, exist_ok=True)

                    # Construct a path for saving the JAMS data for the current tempo
                    tempo_split_path = os.path.join(tempo_split_dir, f'{i}.jams')

                    # Save the sliced JAMS data
                    jam_slice.save(tempo_split_path)

            # Add all tempo changes to the top-level JAMS data
            jam.annotations.append(all_tempo_data)

            # Construct a path for saving the top-level JAMS data
            jams_path = os.path.join(jams_dir, f'{track_label}.jams')

            # Save the JAMS data
            jam.save(jams_path)


if __name__ == '__main__':
    # Construct a path to the base directory
    gpro_dir = 'path/to/DadaGP'

    # Remove existing JAMS files
    clean_jams(gpro_dir)

    # Search the specified directory for valid GuitarPro files
    tracked_files, tracked_dirs = get_valid_files(gpro_dir)

    fw = open('error_files.txt', 'w')

    # Loop through the tracked GuitarPro files
    for gpro_file, dir in zip(tracked_files, tracked_dirs):
        print(f'Processing track \'{gpro_file}\'...')

        # Construct a path to GuitarPro file and JAMS output directory
        gpro_path = os.path.join(dir, gpro_file)
        jams_dir = os.path.join(dir, gpro_file.replace('.', ' - '))

        # Perform the conversion
        try:
            write_jams_guitarpro(gpro_path, jams_dir)
        except:
            print(f'error_track: \'{gpro_file}\'...')
            fw.write(gpro_path + '\n')

    fw.close()
