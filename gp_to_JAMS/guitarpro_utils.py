# Author: Frank Cwitkowitz <fcwitkow@ur.rochester.edu>

from guitarpro import NoteType, Duration, Velocities
from copy import deepcopy

import warnings
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

    def __init__(self, fret, onset, velocity, duration, string):
        """
        Initialize a guitar note.

        Parameters
        ----------
        fret : int
          Fret the note was played on
        onset : int
          Time of the beginning of the note in ticks
        velocity : int
          Velocity of the note
        duration : int
          Amount of ticks after the onset where the note is still active
        string : int (Optional)
          Index of the string the note was played on
        """

        self.fret = fret
        self.onset = onset
        self.velocity = velocity
        self.duration = duration
        self.string = string

        self.effects = []

    def set_duration(self, new_duration):
        """
        Update the duration of the note in ticks.

        Parameters
        ----------
        duration : int
          Amount of ticks the note is active
        """

        self.duration = new_duration

    def parseNoteEffect(self, effect, onset, duration):
        """
        Parse effect attributes relevant for our purposes.

        Parameters
        ----------
        effect : guitarpro.NoteEffect
          Technique and effect information for a note
        """

        # TODO - Add helpful comments and hints to this function

        # Create a new dictionary to hold any effects
        effects = dict()

        # Encode boolean attributes...
        if effect.accentuatedNote:
            effects.update({'accentuated_note' : True})
        if effect.ghostNote:
            effects.update({'ghost_note' : True})
        if effect.hammer:
            # Next note will be a hammer-on or pull-off
            effects.update({'hammer' : True})
        if effect.heavyAccentuatedNote:
            effects.update({'heavy_accentuated_note' : True})
        if effect.letRing:
            effects.update({'let_ring' : True})
        if effect.palmMute:
            effects.update({'palm_mute' : True})
        if effect.staccato:
            effects.update({'staccato' : True})
        if effect.vibrato:
            effects.update({'vibrato' : True})

        if len(effect.slides):
            effects.update({'slide' : [s.name for s in effect.slides]})

        if effect.isBend:
            # Extract information regarding the bend points
            positions, heights, vibratos = zip(*[(p.position / 12,
                                                  6 * p.value / 12,
                                                  p.vibrato) for p in effect.bend.points])

            effects.update({'bend' : {'type' : effect.bend.type.name,
                                      'points' : {'position' : positions, # % of duration
                                                  'height' : heights, # semitones
                                                  'vibrato' : vibratos}}})

        if effect.isGrace:
            effects.update({'grace' : {'fret' : effect.grace.fret,
                                       'transition' : effect.grace.transition.name,
                                       'velocity' : effect.grace.velocity,
                                       'duration' : effect.grace.durationTime,
                                       'on_beat' : effect.grace.isOnBeat,
                                       'dead' : effect.grace.isDead}})

        if effect.isHarmonic:
            if effect.harmonic.type == 1:
                harmonic = {'type' : 'natural'}
            elif effect.harmonic.type == 2:
                harmonic = {'type' : 'artificial'}
                if effect.harmonic.pitch is not None:
                    harmonic.update({'pitch' : {'just' : effect.harmonic.pitch.just,
                                                'accidental' : effect.harmonic.pitch.accidental,
                                                'value' : effect.harmonic.pitch.value,
                                                'intonation' : effect.harmonic.pitch.intonation}})
                if effect.harmonic.octave is not None:
                    harmonic.update({'octave' : effect.harmonic.octave.name})
            elif effect.harmonic.type == 3:
                harmonic = {'type' : 'tapped'}
                if effect.harmonic.fret is not None:
                    harmonic.update({'fret' : effect.harmonic.fret})
            elif effect.harmonic.type == 4:
                harmonic = {'type' : 'pinch'}
            else:
                harmonic = {'type' : 'semi'}

            effects.update({'harmonic' : harmonic})

        # TODO - the following durations might be for whole duration of effect,
        #        not individual notes - may need to factor in tuplet information

        if effect.isTremoloPicking:
            effects.update({'tremolo' : {'duration' : effect.tremoloPicking.duration.time}})

        if effect.isTrill:
            effects.update({'trill' : {'fret' : effect.trill.fret,
                                       'duration' : effect.trill.duration.time}})

        if len(effects):
            # Add onset and duration of the tracked effects
            effects.update({'time' : onset, 'duration' : duration})
            # Add the effects to the note
            self.effects.append(effects)


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

        if gpro_note.type == NoteType.rest:
            # Nothing to do for rests
            return

        # Extract the string, fret, and velocity of the note
        string_idx, fret, velocity = gpro_note.string, gpro_note.value, gpro_note.velocity

        # Make sure a velocity is set for the note
        velocity = max(velocity, Velocities.minVelocity)

        # Scale the duration by the duration percentage
        duration = round(duration * gpro_note.durationPercent)

        # Create a note object to keep track of the GuitarPro note
        note = Note(fret, onset, velocity, duration, string_idx)

        # Parse the relevant techniques and effects
        note.parseNoteEffect(gpro_note.effect, onset, duration)

        if gpro_note.type == NoteType.dead:
            if len(note.effects):
                # Add to existing effects
                note.effects[0].update({'dead_note' : True})
            else:
                # Add as a new effects entry
                note.effects.append({'time' : onset,
                                     'duration' : duration,
                                     'dead_note' : True})

        if gpro_note.type == NoteType.tie:
            # Obtain the last note that occurred on the string
            last_gpro_note = self.gpro_notes[string_idx][-1] \
                             if len(self.gpro_notes[string_idx]) else None
            # Determine if the last note should be extended
            if last_gpro_note is not None:
                # Determine how much to extend the note
                new_duration = onset - last_gpro_note.onset + duration
                # Extend the previous note by the current beat's duration
                last_gpro_note.set_duration(new_duration)
                # Parse any effects on the tie and add them to the last note
                last_gpro_note.parseNoteEffect(gpro_note.effect, onset, duration)
            else:
                warnings.warn('No last note for tie...', RuntimeWarning)
        else:
            # Add the new note to the dictionary under the respective string
            self.gpro_notes[string_idx].append(note)

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
                value = {'fret' : n.fret, 'velocity' : n.velocity}

                if len(n.effects):
                    # Add any note effects to the dictionary
                    value.update({'effects' : n.effects})

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

        # Keep track of the amount of ticks processed within the measure
        measure_ticks = [0.] * len(measure.voices)

        # Loop through voices within the measure
        for v, voice in enumerate(measure.voices):
            # Loop through the beat divisions of the measure
            for beat in voice.beats:
                if total_ticks is None:
                    # Set the current tick to the measure start
                    total_ticks = float(measure.start)

                # Compute the current tick within the measure
                current_tick = total_ticks + measure_ticks[v]

                # Check if there are any tempo changes
                if beat.effect.mixTableChange is not None:
                    if beat.effect.mixTableChange.tempo is not None:
                        # Extract the updated tempo
                        new_tempo = beat.effect.mixTableChange.tempo.value
                        # Update the tempo of the note tracker
                        note_tracker.set_current_tempo(new_tempo)
                        # Add the tempo change to the tracked list
                        tempo_changes.append((current_tick, new_tempo))

                # Obtain the note duration in ticks
                duration_ticks = float(beat.duration.time)

                # Loop through the notes in the beat division
                for note in beat.notes:
                    # Add the note to the tracker
                    note_tracker.track_note(note, current_tick, duration_ticks)

                # Accumulate the ticks of the beat
                measure_ticks[v] += duration_ticks

        # Add the measure ticks to the total ticks
        total_ticks += measure_ticks[0]
        # Check if all ticks were counted
        if measure_ticks[0] < measure.length:
            # Compute the number of ticks missing
            remaining_ticks = measure.length - measure_ticks[0]
            # Add the remaining measure ticks
            total_ticks += remaining_ticks

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
