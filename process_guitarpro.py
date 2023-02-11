# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

# My imports
from guitarpro_utils import validate_gpro_track, parse_notes_gpro_track

# Regular imports
import guitarpro
import os


VALID_GP_EXTS = ['.gp3', '.gp4', '.gp5']
INVALID_EXTS = ['.pygp', '.gp2tokens2gp']


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
    tracked_paths : list of str
      List of paths corresponding to tracked files
    """

    # Keep track of valid GuitarPro files
    tracked_paths, tracked_files = list(), list()

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
        tracked_paths += [dir_path] * len(valid_files)

    return tracked_files, tracked_paths


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
        # Make sure the JAMS directory exists
        os.makedirs(jams_dir, exist_ok=True)

        # Make sure the GuitarPro file can be processed for symbolic datasets
        if validate_gpro_track(gpro_track):
            # Extract notes from the track, given the listed tempo
            note_tracker = parse_notes_gpro_track(gpro_track, gpro_data.tempo)

            # Write the note predictions to a JAMS file
            jam = note_tracker.write_jams()

            # Construct a path for saving the JAMS data
            jams_path = os.path.join(jams_dir, gpro_track.name + '.jams')

            # TODO - is there any metadata to write here?
            # TODO - write instrument here?

            # Save as a JAMS file
            jam.save(jams_path)


if __name__ == '__main__':
    # Construct a path to the base directory
    #gpro_dir = 'path/to/DadaGP'
    gpro_dir = '/home/rockstar/Desktop/Datasets/DadaGP'

    # Search the specified path for valid GuitarPro files
    tracked_files, tracked_paths = get_valid_files(gpro_dir)

    # Construct a path to the JAMS directory
    jams_dir = os.path.join(gpro_dir, 'jams_notes')

    # Loop through the tracked GuitarPro files
    for gpro_file, gpro_path in zip(tracked_files, tracked_paths):
        print(f'Processing track \'{gpro_file}\'...')

        # Construct a path to GuitarPro file and JAMS output directory
        gpro_path_ = os.path.join(gpro_path, gpro_file)
        jams_dir_ = os.path.join(*([jams_dir] + gpro_file.split('.')))

        # Perform the conversion
        write_jams_guitarpro(gpro_path_, jams_dir_)
