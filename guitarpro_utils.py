# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from guitarpro import NoteType, Duration
from copy import deepcopy

import jams


# Add our custom schema for guitar notes
jams.schema.add_namespace('note_tab.json')


def ticks_to_seconds(ticks, tempo):
    """
    Convert an amount of ticks to a concrete time.

    Parameters
    ----------
    ticks : int or float
      Amount of ticks
    tempo : int or float
      Number of beats per minute

    Returns
    ----------
    time : float
      Time in seconds corresponding to number of ticks
    """

    # Number of seconds per beat times number of quarter beats
    time = (60 / tempo) * ticks / Duration.quarterTime

    return time


class Note(object):
    """
    Simple class representing a guitar note for use during GuitarPro file processing.
    """

    def __init__(self, fret, onset, duration, string):
        """
        Initialize a guitar note.

        Parameters
        ----------
        fret : int
          Fret the note was played on
        onset : int
          Time of the beginning of the note in ticks
        duration : int
          Amount of ticks after the onset where the note is still active
        string : int (Optional)
          Index of the string the note was played on
        """

        self.fret = fret
        self.onset = onset
        self.duration = duration
        self.string = string

        # TODO - specify techniques/effects somehow

    def set_duration(self, new_duration):
        """
        Update the duration of the note in ticks.

        Parameters
        ----------
        duration : int
          Amount of ticks the note is active
        """

        self.duration = new_duration


class NoteTracker(object):
    """
    Simple class to keep track of state while tracking notes in a GuitarPro file.
    """

    def __init__(self, default_tempo, tuning):
        """
        Initialize the state of the tracker.

        Parameters
        ----------
        default_tempo : int or float
          Underlying tempo of the track
        tuning : list of guitarpro.GuitarString
            MIDI pitch of each open-string
        """

        # Keep track of both the underlying and current tempo
        self.default_tempo = default_tempo
        self.current_tempo = default_tempo

        # Determine the string indices and open tuning for all strings of the track
        self.string_idcs, self.open_tuning = zip(*[(s.number, s.value) for s in tuning])

        # Initialize a dictionary to hold all notes
        self.gpro_notes = {s : list() for s in self.string_idcs}

    def set_current_tempo(self, tempo=None):
        """
        Update the currently tracked tempo.

        Parameters
        ----------
        tempo : int or float (Optional)
          New tempo
        """

        if tempo is None:
            # Reset the current tempo to the default
            self.current_tempo = self.default_tempo
        else:
            # Update the current tempo
            self.current_tempo = tempo

    def get_current_tempo(self):
        """
        Obtain the currently tracked tempo.

        Returns
        ----------
        tempo : int or float
          Current tracked tempo
        """

        tempo = self.current_tempo

        return tempo

    def track_note(self, gpro_note, onset, duration):
        """
        Update the currently tracked tempo.

        Parameters
        ----------
        gpro_note : guitarpro.Note
          GuitarPro note information
        onset : int
          Time the note begins in ticks
        duration : int
          Amount of ticks the note is active
        """

        # Extract note's string and fret
        string_idx, fret = gpro_note.string, gpro_note.value

        # Scale the duration by the duration percentage
        duration = round(duration * gpro_note.durationPercent)

        # TODO - extract information from NoteEffect and maybe BeatEffect (not MixTableChange)

        # Create a note object to keep track of the GuitarPro note
        note = Note(fret, onset, duration, string_idx)

        if gpro_note.type == NoteType.normal:
            # Add the new note to the dictionary under the respective string
            self.gpro_notes[string_idx].append(note)
        elif gpro_note.type == NoteType.tie:
            # Obtain the last note that occurred on the string
            last_gpro_note = self.gpro_notes[string_idx][-1] \
                             if len(self.gpro_notes[string_idx]) else None
            # Determine if the last note should be extended
            if last_gpro_note is not None:
                # Determine how much to extend the note
                new_duration = onset - last_gpro_note.onset + duration
                # Extend the previous note by the current beat's duration
                last_gpro_note.set_duration(new_duration)
        elif gpro_note.type == NoteType.dead:
            # TODO - dead note
            pass
        else:
            # TODO - rest - don't track note
            pass

    def write_jams(self):
        """
        Write the tracked note data to a JAMS file.

        Returns
        ----------
        jam : JAMS object
          JAMS file data
        """

        # Create a new JAMS object
        jam = jams.JAMS()

        # Loop through all tracked strings
        for s, p in zip(self.string_idcs, self.open_tuning):
            # Create a new annotation for guitar tablature
            string_data = jams.Annotation(namespace='note_tab')
            # Set the source (string) and tuning for the string
            string_data.sandbox.update(string_index=s, open_tuning=p)
            # Loop through all notes
            for n in self.gpro_notes[s]:
                # Dictionary of tablature note attributes
                value = {
                    'fret' : n.fret,
                    # TODO - additional fields here
                }
                # Add an annotation for the note
                string_data.append(time=n.onset, duration=n.duration, value=value)
            # Add the annotation to the JAMS object
            jam.annotations.append(string_data)

        return jam


VALID_INSTRUMENTS = {
    24 : 'Acoustic Nylon Guitar',
    25 : 'Acoustic Steel Guitar',
    26 : 'Electric Jazz Guitar',
    27 : 'Electric Clean Guitar',
    28 : 'Electric Muted Guitar',
    29 : 'Overdriven Guitar',
    30 : 'Distortion Guitar',
    31 : 'Guitar Harmonics',
    32 : 'Acoustic Bass',
    33 : 'Fingered Electric Bass',
    34 : 'Plucked Electric Bass',
    35 : 'Fretless Bass',
    36 : 'Slap Bass 1',
    37 : 'Slap Bass 2',
    38 : 'Synth Bass 1',
    39 : 'Synth Bass 2',
    105 : 'Banjo'
}


def validate_gpro_track(gpro_track):
    """
    Helper function to determine which GuitarPro tracks are valid for our purposes.

    Parameters
    ----------
    gpro_track : guitarpro.Track
      GuitarPro track to validate

    Returns
    ----------
    valid : bool
      Whether the GuitarPro track is considered valid
    """

    # Determine if this is a percussive track
    is_percussive = gpro_track.isPercussionTrack

    # Determine if this is a valid guitar track
    is_guitar = (24 <= gpro_track.channel.instrument <= 31)

    # TODO - support guitar fret noise (120)?

    # Determine if this is a valid bass track
    is_bass = (32 <= gpro_track.channel.instrument <= 39)

    # Determine if this is a valid banjo track
    is_banjo = gpro_track.isBanjoTrack or gpro_track.channel.instrument == 105

    # Determine if the track is valid
    valid = not is_percussive and (is_guitar or is_bass or is_banjo)

    return valid


def parse_notes_gpro_track(gpro_track, default_tempo):
    """
    Track duration and MIDI notes spread across strings within a GuitarPro track.

    Parameters
    ----------
    gpro_track : guitarpro.Track
      GuitarPro track data
    default_tempo : int
      Track tempo for inferring note onset and duration

    Returns
    ----------
    note_tracker : NoteTracker
      Tracking object to maintain notes and their attributes
    tempo_changes : list of (tick, tempo) tuples
      Collection of tempo changes along with the tick where they occurred
    """

    # Make a copy of the track, so that it can be modified without consequence
    gpro_track = deepcopy(gpro_track)

    # Initialize a tracker to keep track of GuitarPro notes
    note_tracker = NoteTracker(default_tempo, gpro_track.strings)

    # Keep track of cumulative ticks
    total_time = None
    total_ticks = None

    # Initialize a dictionary for tempo changes
    tempo_changes = [(0, default_tempo)]

    # Determine how many measures are in the track
    total_num_measures = len(gpro_track.measures)

    # Keep track of the current measure
    current_measure = 0
    # Keep track of the last measure which opened a repeat
    repeat_measure = 0
    # Keep track of the measure after the most recently encountered repeat close
    next_jump = None

    # Initialize a counter to keep track of how many times a repeat was obeyed
    repeat_count = 0

    # Loop through the track's measures
    while current_measure < total_num_measures:
        # Process the current measure
        measure = gpro_track.measures[current_measure]

        if measure.header.repeatAlternative != 0:
            # The 'repeatAlternative' attribute seems to be encoded as binary a binary vector,
            # where the integers k in the measure header represent a 1 in the kth digit
            alt_repeat_num = sum([2 ** k for k in range(repeat_count)])
            # Check if it is time to jump past the repeat close
            if alt_repeat_num >= measure.header.repeatAlternative:
                # Reset the external repeat counter
                repeat_count = 0
                # Indicate that this measure should always cause a jump from now on
                measure.header.repeatAlternative = -1
                # Jump past the repeat
                current_measure = next_jump
                continue

        if measure.isRepeatOpen:
            # Jump back to this measure at the next repeat close
            repeat_measure = current_measure

        # Keep track of the amount of time processed within the measure
        measure_ticks = [0] * len(measure.voices)
        measure_time = [0] * len(measure.voices)

        # Loop through voices within the measure
        for v, voice in enumerate(measure.voices):
            # Loop through the beat divisions of the measure
            for beat in voice.beats:
                if total_ticks is None or total_time is None:
                    # Set the current time to the start of the measure
                    total_ticks = measure.start
                    total_time = ticks_to_seconds(measure.start,
                                                  note_tracker.get_current_tempo())

                # Check if there are any tempo changes
                if beat.effect.mixTableChange is not None:
                    if beat.effect.mixTableChange.tempo is not None:
                        # Extract the updated tempo
                        new_tempo = beat.effect.mixTableChange.tempo.value
                        # Update the tempo of the note tracker
                        note_tracker.set_current_tempo(new_tempo)
                        # Add the tempo change to the tracked list
                        tempo_changes.append((total_ticks + measure_ticks[v], new_tempo))

                # Obtain the note duration in ticks and convert from ticks to seconds
                duration_ticks = beat.duration.time
                duration_seconds = ticks_to_seconds(duration_ticks,
                                                    note_tracker.get_current_tempo())

                # Loop through the notes in the beat division
                for note in beat.notes:
                    # Add the note to the tracker
                    note_tracker.track_note(note, total_ticks + measure_ticks[v], duration_ticks)

                # Accumulate the time of the beat
                measure_ticks[v] += duration_ticks
                measure_time[v] += duration_seconds

        # Add the measure time to the current accumulated time
        total_ticks += measure_ticks[0]
        total_time += measure_time[0]
        # Check if all ticks were counted
        if measure_ticks[0] != measure.length:
            # Compute the number of ticks missing
            remaining_ticks = measure.length - measure_ticks[0]
            # Add the time for the missing ticks
            total_ticks += remaining_ticks
            total_time += ticks_to_seconds(remaining_ticks,
                                           note_tracker.get_current_tempo())

        if measure.repeatClose > 0:
            # Set the (alternate repeat) jump to the next measure
            next_jump = current_measure + 1
            # Jump back to where the repeat begins
            current_measure = repeat_measure
            # Decrement the measure's repeat counter
            measure.repeatClose -= 1
            # Increment the external repeat counter
            repeat_count += 1
        else:
            if measure.repeatClose == 0:
                # Reset the external repeat counter
                repeat_count = 0
            # Increment the measure pointer
            current_measure += 1

    # Add the final tick so duration can be inferred
    tempo_changes.append((total_ticks, -1))

    return note_tracker, tempo_changes
