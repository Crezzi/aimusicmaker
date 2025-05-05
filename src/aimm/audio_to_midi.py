"""
Possible modules:

basic-pitch
librosa
mido
audio-to-midi
sound_to_midi
moviepy
wave
midiutil.MIDIFile
scipy.io.wavefile
"""

"""
Polyphonic Audio to MIDI Converter

This script converts polyphonic audio files (songs with multiple instruments/notes)
to MIDI by extracting multiple concurrent notes. It uses librosa for audio analysis
and midiutil for MIDI file creation, with additional techniques for handling polyphony.

Usage:
    python polyphonic_audio_to_midi.py input_audio.wav output_midi.mid

Requirements:
    - librosa
    - numpy
    - midiutil
    - scipy
"""



import os
import argparse
import numpy as np
import librosa
from midiutil import MIDIFile
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import basic_pitch


def hz_to_midi_note(frequency):
    """Convert frequency in Hz to MIDI note number"""
    if frequency <= 0:
        return 0
    return int(round(12 * np.log2(frequency / 440) + 69))


def midi_note_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz"""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def extract_notes_from_cqt(cqt_magnitudes, threshold_percentile=97, min_duration_frames=2):
    """
    Extract multiple concurrent notes from a CQT spectrogram
    
    Parameters:
    -----------
    cqt_magnitudes : np.ndarray
        CQT magnitude spectrogram
    threshold_percentile : float
        Percentile to use as threshold for detecting peaks
    min_duration_frames : int
        Minimum duration of notes in frames
    
    Returns:
    --------
    note_events : list of tuples
        List of (note, start_frame, end_frame) tuples
    """
    # Normalize CQT
    cqt_norm = librosa.util.normalize(cqt_magnitudes, axis=0)
    
    # Threshold for peak detection (dynamic threshold using percentile)
    threshold = np.percentile(cqt_norm, threshold_percentile)
    
    # Initialize note tracking
    active_notes = {}  # Map from note to start frame
    note_events = []  # List of (note, start_frame, end_frame) tuples
    
    # Process each frame
    for frame_idx in range(cqt_norm.shape[1]):
        # Find peaks in the current frame
        peaks, _ = find_peaks(cqt_norm[:, frame_idx], height=threshold, distance=1)
        
        current_notes = set()
        
        # Process detected peaks (potential notes)
        for peak_idx in peaks:
            midi_note = peak_idx + 24  # CQT usually starts around MIDI note 24 (C1)
            current_notes.add(midi_note)
            
            # Note onset detection
            if midi_note not in active_notes:
                active_notes[midi_note] = frame_idx
        
        # Note offset detection (find notes that were active but are no longer detected)
        notes_to_remove = []
        for note in active_notes:
            if note not in current_notes:
                start_frame = active_notes[note]
                end_frame = frame_idx
                
                # Only add notes that meet minimum duration
                if end_frame - start_frame >= min_duration_frames:
                    note_events.append((note, start_frame, end_frame))
                
                notes_to_remove.append(note)
        
        # Remove ended notes
        for note in notes_to_remove:
            del active_notes[note]
    
    # Handle notes still active at the end
    for note, start_frame in active_notes.items():
        end_frame = cqt_norm.shape[1] - 1
        if end_frame - start_frame >= min_duration_frames:
            note_events.append((note, start_frame, end_frame))
    
    return note_events


def process_audio_harmonics(y, sr, fmin=65.0, n_octaves=6, bins_per_octave=36, 
                           threshold_percentile=97, min_duration_ms=100):
    """
    Process audio using harmonic CQT to extract notes
    
    Parameters:
    -----------
    y : np.ndarray
        Audio time series
    sr : int
        Sample rate
    fmin : float
        Minimum frequency
    n_octaves : int
        Number of octaves in the CQT
    bins_per_octave : int
        Number of bins per octave in the CQT
    threshold_percentile : float
        Percentile threshold for peak detection
    min_duration_ms : float
        Minimum note duration in milliseconds
    
    Returns:
    --------
    note_events : list of tuples
        List of (midi_note, start_time, end_time) tuples
    """
    # Calculate the hop length based on typical frame rate
    hop_length = 512
    
    # Compute harmonic CQT
    # The harmonic CQT helps separate the harmonic content
    harmonic, percussive = librosa.effects.hpss(y)
    C = np.abs(librosa.cqt(harmonic, sr=sr, hop_length=hop_length,
                          fmin=fmin, n_bins=n_octaves * bins_per_octave,
                          bins_per_octave=bins_per_octave))
    
    # Convert minimum duration from ms to frames
    min_duration_frames = int(min_duration_ms / 1000 * sr / hop_length)
    
    # Extract note events as (note, start_frame, end_frame)
    note_events_frames = extract_notes_from_cqt(
        C, threshold_percentile=threshold_percentile, 
        min_duration_frames=min_duration_frames
    )
    
    # Convert frames to time
    frame_to_time = lambda frame: frame * hop_length / sr
    note_events = [(note, frame_to_time(start), frame_to_time(end)) 
                  for note, start, end in note_events_frames]
    
    return note_events, C


def polyphonic_audio_to_midi(input_file, output_file, threshold_percentile=97,
                            min_duration_ms=100, velocity_scale=100, 
                            tempo=120, visualize=False):
    """
    Convert a polyphonic audio file to MIDI
    
    Parameters:
    -----------
    input_file : str
        Path to the input audio file
    output_file : str
        Path to the output MIDI file
    threshold_percentile : float
        Percentile threshold for note detection
    min_duration_ms : float
        Minimum note duration in milliseconds
    velocity_scale : int
        Base MIDI velocity (will be adjusted by note intensity)
    tempo : int
        Tempo in BPM
    visualize : bool
        Whether to visualize the CQT and detected notes
    """
    print(f"Converting polyphonic audio: {input_file} to {output_file}")
    
    # Load the audio file
    print("Loading audio file...")
    y, sr = librosa.load(input_file, sr=None)
    
    # Process audio to extract notes with timing information
    print("Extracting notes from audio...")
    note_events, cqt_data = process_audio_harmonics(
        y, sr, 
        threshold_percentile=threshold_percentile,
        min_duration_ms=min_duration_ms
    )
    
    print(f"Detected {len(note_events)} notes")
    
    # Create a MIDI file
    midi = MIDIFile(1)  # One track
    track = 0
    time = 0
    
    # Set tempo
    midi.addTempo(track, time, tempo)
    
    # Add all detected notes to the MIDI file
    for note, start_time, end_time in note_events:
        # Convert time to beats
        start_beats = start_time * (tempo / 60)
        duration_beats = (end_time - start_time) * (tempo / 60)
        
        # Add the note (channel 0 is usually piano)
        # Adjust velocity based on position in valid MIDI range (0-127)
        velocity = min(127, max(1, int(velocity_scale)))
        midi.addNote(track, 0, int(note), start_beats, duration_beats, velocity)
    
    # Write the MIDI file
    with open(output_file, "wb") as f:
        midi.writeFile(f)
    
    print(f"Successfully converted audio to MIDI: {output_file}")
    
    # Optional visualization
    if visualize:
        plt.figure(figsize=(12, 8))
        
        # Plot CQT
        librosa.display.specshow(
            librosa.amplitude_to_db(cqt_data, ref=np.max),
            sr=sr, x_axis='time', y_axis='cqt_note',
            hop_length=512, bins_per_octave=36
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Constant-Q Power Spectrum')
        
        # Save the visualization
        vis_path = os.path.splitext(output_file)[0] + '_visualization.png'
        plt.savefig(vis_path)
        print(f"Visualization saved to {vis_path}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Convert polyphonic audio to MIDI")
    parser.add_argument("input_file", help="Input audio file path")
    parser.add_argument("output_file", help="Output MIDI file path")
    parser.add_argument("--threshold", type=float, default=97,
                        help="Percentile threshold for note detection (0-100)")
    parser.add_argument("--min-duration", type=float, default=100,
                        help="Minimum note duration in milliseconds")
    parser.add_argument("--velocity", type=int, default=80,
                        help="Base MIDI velocity (0-127)")
    parser.add_argument("--tempo", type=int, default=120,
                        help="Tempo in BPM")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization of the CQT and detected notes")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.isfile(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Convert audio to MIDI
    polyphonic_audio_to_midi(
        args.input_file,
        args.output_file,
        threshold_percentile=args.threshold,
        min_duration_ms=args.min_duration,
        velocity_scale=args.velocity,
        tempo=args.tempo,
        visualize=args.visualize
    )


if __name__ == "__main__":
    main()