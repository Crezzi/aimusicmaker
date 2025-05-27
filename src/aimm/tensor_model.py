import os
import numpy as np
import pretty_midi
import tf_keras
from tf_keras import layers
import glob
import random
from typing import List, Tuple, Optional, Union, Any
from collections import defaultdict


def load_midi_file(file_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Load a MIDI file and return a PrettyMIDI object.
    Returns None if the file can't be loaded.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        print(f"Loaded {file_path}: {len(midi_data.instruments)} instruments, "
              f"{midi_data.get_end_time():.2f}s duration")
        return midi_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_polyphonic_events(midi_data: pretty_midi.PrettyMIDI, 
                             instrument_index: int = 0, 
                             time_step: float = 0.125) -> List[Tuple[List[int], float]]:
    """
    Extract polyphonic events from MIDI data by discretizing time into steps.
    
    This function creates "events" where each event contains all notes that are
    active at that time step, allowing for chord detection and generation.
    
    Args:
        midi_data: PrettyMIDI object
        instrument_index: Index of the instrument to extract notes from
        time_step: Time resolution in seconds (0.125 = 1/8 note at 120 BPM)
        
    Returns:
        List of (active_pitches, duration_until_next_event) tuples
    """
    if len(midi_data.instruments) == 0:
        print("WARNING: No instruments found in MIDI file")
        return []
    
    if instrument_index >= len(midi_data.instruments):
        print(f"WARNING: Instrument index {instrument_index} out of range. Using instrument 0.")
        instrument_index = 0
    
    instrument = midi_data.instruments[instrument_index]
    
    # Get the total duration and create time grid
    end_time = midi_data.get_end_time()
    time_steps = np.arange(0, end_time, time_step)
    
    events = []
    
    print(f"Processing {len(time_steps)} time steps with resolution {time_step}s")
    
    for i, current_time in enumerate(time_steps):
        # Find all notes that are active at this time step
        active_pitches = []
        
        for note in instrument.notes:
            if note.start <= current_time < note.end:
                active_pitches.append(note.pitch)
        
        # Remove duplicates and sort
        active_pitches = sorted(list(set(active_pitches)))
        
        # Calculate duration until next event (or use time_step as default)
        if i < len(time_steps) - 1:
            duration = time_steps[i + 1] - current_time
        else:
            duration = time_step
        
        events.append((active_pitches, duration))
    
    # Filter out empty events (silence) but keep some for musical phrasing
    filtered_events = []
    silence_count = 0
    
    for pitches, duration in events:
        if len(pitches) > 0:
            # Add accumulated silence before this chord if any
            if silence_count > 0:
                filtered_events.append(([], silence_count * time_step))
                silence_count = 0
            filtered_events.append((pitches, duration))
        else:
            silence_count += 1
            # Don't let silence accumulate too much
            if silence_count >= 8:  # Max 1 second of silence at 0.125s resolution
                filtered_events.append(([], silence_count * time_step))
                silence_count = 0
    
    # Add final silence if any
    if silence_count > 0:
        filtered_events.append(([], silence_count * time_step))
    
    # Statistics
    chord_events = [e for e in filtered_events if len(e[0]) > 1]
    single_note_events = [e for e in filtered_events if len(e[0]) == 1]
    silence_events = [e for e in filtered_events if len(e[0]) == 0]
    
    print(f"Extracted {len(filtered_events)} events:")
    print(f"  - {len(chord_events)} chords (2+ notes)")
    print(f"  - {len(single_note_events)} single notes")
    print(f"  - {len(silence_events)} silence periods")
    
    if chord_events:
        chord_sizes = [len(e[0]) for e in chord_events]
        print(f"  - Chord sizes: {min(chord_sizes)}-{max(chord_sizes)} notes, avg: {np.mean(chord_sizes):.1f}")
    
    return filtered_events


def encode_polyphonic_events(events: List[Tuple[List[int], float]], max_notes: int = 10) -> Tuple[np.ndarray, List[float]]:
    """
    Encode polyphonic events into a format suitable for neural network training.
    
    Each event is encoded as a binary vector of length 128 (for MIDI pitches 0-127)
    where 1 indicates the note is active and 0 indicates it's not.
    
    Args:
        events: List of (active_pitches, duration) tuples
        max_notes: Maximum number of simultaneous notes to consider (for filtering)
        
    Returns:
        Tuple of (encoded_events, duration_data)
    """
    encoded = []
    durations = []
    
    for pitches, duration in events:
        # Filter out events with too many simultaneous notes (likely errors)
        if len(pitches) > max_notes:
            print(f"WARNING: Skipping event with {len(pitches)} simultaneous notes (max: {max_notes})")
            continue
        
        # Create binary encoding
        encoding = np.zeros(128, dtype=np.float32)
        for pitch in pitches:
            if 0 <= pitch <= 127:
                encoding[pitch] = 1.0
        
        encoded.append(encoding)
        durations.append(duration)
    
    return np.array(encoded), durations


def create_polyphonic_sequences(encoded_events: np.ndarray, 
                               durations: List[float], 
                               sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Create sequences for training the polyphonic model.
    
    Args:
        encoded_events: Binary encoded events (n_events, 128)
        durations: Duration for each event
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (input_sequences, output_events, duration_data)
    """
    if len(encoded_events) < sequence_length + 1:
        raise ValueError(f"Not enough events ({len(encoded_events)}) to create sequences of length {sequence_length}")
    
    network_input = []
    network_output = []
    
    # Create overlapping sequences
    for i in range(len(encoded_events) - sequence_length):
        sequence_in = encoded_events[i:i + sequence_length]
        sequence_out = encoded_events[i + sequence_length]
        
        network_input.append(sequence_in)
        network_output.append(sequence_out)
    
    network_input = np.array(network_input)
    network_output = np.array(network_output)
    
    print(f"Created {len(network_input)} training sequences")
    print(f"Input shape: {network_input.shape}, Output shape: {network_output.shape}")
    
    return network_input, network_output, durations


def create_polyphonic_model(sequence_length: int = 20) -> tf_keras.Sequential:
    """
    Create an LSTM model for polyphonic MIDI generation.
    
    This model outputs a 128-dimensional binary vector representing
    which MIDI notes should be active at each time step.
    
    Args:
        sequence_length: Length of input sequences
        
    Returns:
        Compiled tf_keras model
    """
    model = tf_keras.Sequential()
    
    # Input shape: (batch_size, sequence_length, 128)
    model.add(layers.LSTM(512, input_shape=(sequence_length, 128), return_sequences=True))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.LSTM(512))
    model.add(layers.Dropout(0.3))
    
    # Dense layers for better pattern learning
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    
    # Output layer: 128 neurons with sigmoid activation for binary classification
    # Each neuron represents whether a MIDI note should be active
    model.add(layers.Dense(128, activation='sigmoid'))
    
    # Use binary crossentropy since each output is an independent binary decision
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['binary_accuracy']
    )
    
    print("Polyphonic model architecture:")
    model.summary()
    
    return model


def train_polyphonic_model(midi_files: List[str], 
                          sequence_length: int = 20, 
                          epochs: int = 50, 
                          batch_size: int = 32) -> Tuple[tf_keras.Sequential, np.ndarray, List[float]]:
    """
    Train the polyphonic model on MIDI files.
    
    Args:
        midi_files: List of paths to MIDI files
        sequence_length: Length of sequences for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple of (trained_model, all_encoded_events, duration_data)
    """
    all_events = []
    successful_files = 0
    
    print(f"Processing {len(midi_files)} MIDI files for polyphonic training...")
    
    for i, file in enumerate(midi_files):
        print(f"Processing file {i+1}/{len(midi_files)}: {os.path.basename(file)}")
        midi_data = load_midi_file(file)
        if midi_data:
            events = extract_polyphonic_events(midi_data)
            if events:
                all_events.extend(events)
                successful_files += 1
            else:
                print(f"WARNING: No events extracted from {file}")
    
    if not all_events:
        raise ValueError("No events were successfully extracted from any MIDI files!")
    
    print(f"\nSuccessfully processed {successful_files}/{len(midi_files)} files")
    print(f"Total events extracted: {len(all_events)}")
    
    # Encode events
    encoded_events, duration_data = encode_polyphonic_events(all_events)
    
    # Create training sequences
    network_input, network_output, _ = create_polyphonic_sequences(
        encoded_events, duration_data, sequence_length
    )
    
    print(f"Final training data shape: {network_input.shape}")
    print(f"Starting polyphonic training for {epochs} epochs with batch size {batch_size}")
    
    # Create and train model
    model = create_polyphonic_model(sequence_length)
    
    history = model.fit(
        network_input, 
        network_output, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    # Training statistics
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_acc = history.history['binary_accuracy'][-1]
    
    print(f"\nPolyphonic training completed.")
    print(f"Final loss: {final_loss:.4f}, Final validation loss: {final_val_loss:.4f}")
    print(f"Final binary accuracy: {final_acc:.4f}")
    
    return model, encoded_events, duration_data


def generate_polyphonic_sequence(model: tf_keras.Sequential, 
                                seed_sequence: np.ndarray, 
                                duration_data: List[float],
                                num_events: int = 50, 
                                temperature: float = 1.0,
                                note_threshold: float = 0.3) -> List[Tuple[List[int], float]]:
    """
    Generate a polyphonic sequence using the trained model.
    
    Args:
        model: Trained polyphonic model
        seed_sequence: Initial sequence of encoded events
        duration_data: Duration information from training data
        num_events: Number of events to generate
        temperature: Temperature for sampling (higher = more random)
        note_threshold: Threshold for determining if a note should be active
        
    Returns:
        List of (active_pitches, duration) tuples
    """
    print(f"Generating {num_events} polyphonic events with temperature {temperature}")
    print(f"Using note threshold: {note_threshold}")
    
    if len(seed_sequence) != model.input_shape[1]:
        raise ValueError(f"Seed sequence length ({len(seed_sequence)}) must match model input shape ({model.input_shape[1]})")
    
    pattern = seed_sequence.copy()
    generated_events = []
    
    for i in range(num_events):
        if i % 10 == 0:
            print(f"Generated {i}/{num_events} events...")
        
        # Prepare input
        x = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
        
        # Get prediction
        prediction = model.predict(x, verbose=0)[0]
        
        # DEBUG: Print prediction statistics for first few generations
        if i < 3:
            print(f"Event {i}: Prediction range [{prediction.min():.4f}, {prediction.max():.4f}], "
                  f"Mean: {prediction.mean():.4f}")
            top_5_indices = np.argsort(prediction)[-5:]
            print(f"Top 5 predicted notes: {[(idx, prediction[idx]) for idx in reversed(top_5_indices)]}")
        
        # Apply temperature - fix the temperature application
        if temperature != 1.0 and temperature > 0:
            # For sigmoid outputs, we need a different temperature application
            prediction = np.clip(prediction, 1e-8, 1 - 1e-8)  # Avoid log(0) and log(1)
            logits = np.log(prediction / (1 - prediction))  # Convert sigmoid to logits
            logits = logits / temperature
            prediction = 1 / (1 + np.exp(-logits))  # Convert back to probabilities
        
        # Multiple strategies for note selection to ensure we get some notes
        active_notes = []
        
        # Strategy 1: Direct threshold
        for pitch, prob in enumerate(prediction):
            if prob > note_threshold:
                active_notes.append(pitch)
        
        # Strategy 2: If no notes selected, take the top N most probable
        if len(active_notes) == 0:
            # Take top 1-3 notes based on probability
            num_top_notes = min(3, max(1, int(np.sum(prediction > 0.1))))
            top_indices = np.argsort(prediction)[-num_top_notes:]
            for idx in top_indices:
                if prediction[idx] > 0.05:  # Very low threshold
                    active_notes.append(idx)
        
        # Strategy 3: If still empty, force at least one note
        if len(active_notes) == 0:
            best_note = np.argmax(prediction)
            if prediction[best_note] > 0.01:  # Extremely low threshold
                active_notes.append(best_note)
                print(f"WARNING: Forced note selection at event {i}, note {best_note} with prob {prediction[best_note]:.4f}")
        
        # Ensure we don't have too many simultaneous notes
        if len(active_notes) > 6:  # Limit to 6 simultaneous notes
            # Keep the most probable notes
            note_probs = [(pitch, prediction[pitch]) for pitch in active_notes]
            note_probs.sort(key=lambda x: x[1], reverse=True)
            active_notes = [pitch for pitch, _ in note_probs[:6]]
        
        # Sample duration - ensure we have valid durations
        if duration_data:
            duration = random.choice(duration_data)
        else:
            duration = 0.5  # Default duration
        
        # Ensure minimum duration
        duration = max(duration, 0.1)
        
        generated_events.append((active_notes, duration))
        
        # DEBUG: Print info for first few events
        if i < 5:
            print(f"Event {i}: Generated {len(active_notes)} notes: {active_notes}, duration: {duration:.3f}")
        
        # Update pattern for next prediction
        new_encoding = np.zeros(128, dtype=np.float32)
        for pitch in active_notes:
            if 0 <= pitch < 128:
                new_encoding[pitch] = 1.0
        
        pattern = np.vstack([pattern[1:], new_encoding])
    
    print(f"Generation complete! Generated {len(generated_events)} events")
    
    # Statistics
    chord_events = [e for e in generated_events if len(e[0]) > 1]
    single_note_events = [e for e in generated_events if len(e[0]) == 1]
    silence_events = [e for e in generated_events if len(e[0]) == 0]
    total_notes = sum(len(e[0]) for e in generated_events)
    
    print(f"Generated events breakdown:")
    print(f"  - {len(chord_events)} chords (2+ notes)")
    print(f"  - {len(single_note_events)} single notes")
    print(f"  - {len(silence_events)} silence periods")
    print(f"  - Total notes across all events: {total_notes}")
    
    if total_notes == 0:
        print("ERROR: No notes generated! Check model training and thresholds.")
    
    return generated_events


def create_midi_from_polyphonic_events(events: List[Tuple[List[int], float]], 
                                      tempo: int = 120) -> pretty_midi.PrettyMIDI:
    """
    Create a MIDI file from polyphonic events.
    
    Args:
        events: List of (active_pitches, duration) tuples
        tempo: Tempo in beats per minute
        
    Returns:
        PrettyMIDI object with polyphonic music
    """
    print(f"Creating polyphonic MIDI with {len(events)} events at {tempo} BPM")
    
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    current_time = 0.0
    total_notes_created = 0
    
    for i, (pitches, duration) in enumerate(events):
        # DEBUG: Print first few events
        if i < 5:
            print(f"Event {i}: Creating {len(pitches)} notes at time {current_time:.2f}s, duration {duration:.2f}s")
            if pitches:
                print(f"  Pitches: {pitches}")
        
        # Create all notes that should start at this time
        for pitch in pitches:
            if 0 <= pitch <= 127:
                note = pretty_midi.Note(
                    velocity=80,
                    pitch=int(pitch),
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                total_notes_created += 1
            else:
                print(f"WARNING: Invalid pitch {pitch} at event {i}")
        
        # Advance time
        current_time += duration
    
    midi.instruments.append(instrument)
    
    print(f"Polyphonic MIDI created: {current_time:.2f} seconds total duration")
    print(f"Total notes created: {total_notes_created}")
    
    if total_notes_created == 0:
        print("ERROR: No notes were created! MIDI will be empty.")
        # Let's add a test note to verify MIDI creation works
        test_note = pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0)
        instrument.notes.append(test_note)
        print("Added test note (C4) to prevent empty MIDI")
    
    return midi


def polyphonic_midi_pipeline(midi_folder: str, 
                           output_file: str, 
                           sequence_length: int = 20, 
                           epochs: int = 50, 
                           events_to_generate: int = 100, 
                           temperature: float = 1.0,
                           note_threshold: float = 0.5) -> tf_keras.Sequential:
    """
    Complete pipeline for polyphonic MIDI generation.
    
    Args:
        midi_folder: Path to folder containing MIDI files
        output_file: Path where the generated MIDI will be saved
        sequence_length: Length of sequences for training and generation
        epochs: Number of training epochs
        events_to_generate: Number of events to generate
        temperature: Randomness factor for generation
        note_threshold: Threshold for note activation
        
    Returns:
        Trained tf_keras model
    """
    print("=" * 60)
    print("POLYPHONIC MIDI GENERATION PIPELINE")
    print("=" * 60)
    
    # Find MIDI files
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))
    
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_folder}")
    
    print(f"Found {len(midi_files)} MIDI files in {midi_folder}")
    
    # Train model
    print("\n" + "=" * 30)
    print("TRAINING PHASE")
    print("=" * 30)
    model, all_encoded_events, duration_data = train_polyphonic_model(
        midi_files, sequence_length, epochs
    )
    
    # Generate music
    print("\n" + "=" * 30)
    print("GENERATION PHASE")
    print("=" * 30)
    
    # Get seed sequence
    if len(all_encoded_events) < sequence_length:
        raise ValueError(f"Not enough events in training data ({len(all_encoded_events)}) for sequence length {sequence_length}")
    
    start_index = random.randint(0, len(all_encoded_events) - sequence_length)
    seed_sequence = all_encoded_events[start_index:start_index + sequence_length]
    
    print(f"Using seed sequence starting at index {start_index}")
    
    # Generate polyphonic sequence
    generated_events = generate_polyphonic_sequence(
        model, seed_sequence, duration_data, events_to_generate, temperature, note_threshold
    )
    
    # Create output MIDI
    print("\n" + "=" * 30)
    print("MIDI CREATION PHASE")
    print("=" * 30)
    
    midi_output = create_midi_from_polyphonic_events(generated_events, tempo=120)
    
    # Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    midi_output.write(output_file)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Polyphonic MIDI saved to: {output_file}")
    print(f"Total duration: {midi_output.get_end_time():.2f} seconds")
    print(f"Events generated: {len(generated_events)}")
    print(f"{'='*60}")
    
    return model


# Example usage
if __name__ == "__main__":
    try:
        midi_folder = "data/midi/testing/Cymatics Nebula MIDI Collection/Pop"
        output_file = "data/midi/testing/Cymatics Nebula MIDI Collection/temp/polyphonic_output.mid"
        
        if not os.path.exists(midi_folder):
            print(f"ERROR: MIDI folder does not exist: {midi_folder}")
            exit(1)
        
        model = polyphonic_midi_pipeline(
            midi_folder=midi_folder,
            output_file=output_file,
            sequence_length=20,  # Shorter sequences work better for polyphonic data
            epochs=5,  # Adjust based on your needs
            events_to_generate=100,
            temperature=1.0,
            note_threshold=0.2  # Much lower threshold to ensure notes are generated
        )
        
        print("\nPolyphonic pipeline completed successfully!")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)