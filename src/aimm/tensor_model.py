import os
import numpy as np
import pretty_midi
import tf_keras
from tf_keras import layers
import glob
import random
from typing import List, Tuple, Optional, Union, Any


def load_midi_file(file_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Load a MIDI file and return a PrettyMIDI object.
    Returns None if the file can't be loaded.
    
    Args:
        file_path: Path to the MIDI file
        
    Returns:
        PrettyMIDI object or None if loading fails
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        # DEBUG: Print basic info about loaded MIDI
        print(f"Loaded {file_path}: {len(midi_data.instruments)} instruments, "
              f"{midi_data.get_end_time():.2f}s duration")
        return midi_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_notes(midi_data: pretty_midi.PrettyMIDI, instrument_index: int = 0) -> List[Tuple[int, float, float]]:
    """
    Extract notes from the first instrument of a PrettyMIDI object.
    
    IMPORTANT: This function extracts (pitch, start_time, duration) tuples.
    The duration information is crucial for generating realistic note lengths.
    
    Args:
        midi_data: PrettyMIDI object
        instrument_index: Index of the instrument to extract notes from
        
    Returns:
        List of (pitch, start_time, duration) tuples
    """
    if len(midi_data.instruments) == 0:
        print("WARNING: No instruments found in MIDI file")
        return []
    
    if instrument_index >= len(midi_data.instruments):
        print(f"WARNING: Instrument index {instrument_index} out of range. Using instrument 0.")
        instrument_index = 0
    
    instrument = midi_data.instruments[instrument_index]
    notes = []
    
    # Extract note information including duration
    for note in instrument.notes:
        duration = note.end - note.start
        # FILTER: Skip notes that are too short (likely errors) or too long (likely sustained notes)
        if duration < 0.01:  # Skip notes shorter than 10ms
            print(f"WARNING: Skipping very short note (duration: {duration:.4f}s)")
            continue
        if duration > 10.0:  # Skip notes longer than 10 seconds
            print(f"WARNING: Skipping very long note (duration: {duration:.2f}s)")
            continue
            
        notes.append((note.pitch, note.start, duration))
    
    # Sort by start time to maintain temporal order
    sorted_notes = sorted(notes, key=lambda x: x[1])
    
    # DEBUG: Print statistics about extracted notes
    if sorted_notes:
        durations = [note[2] for note in sorted_notes]
        pitches = [note[0] for note in sorted_notes]
        print(f"Extracted {len(sorted_notes)} notes. "
              f"Pitch range: {min(pitches)}-{max(pitches)}, "
              f"Duration range: {min(durations):.3f}-{max(durations):.3f}s, "
              f"Avg duration: {np.mean(durations):.3f}s")
    
    return sorted_notes


def create_sequences_with_duration(notes: List[Tuple[int, float, float]], sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Create sequences of notes for training, preserving duration information.
    
    CRITICAL: This function now preserves duration data alongside pitch sequences.
    The duration_data will be used later to generate notes with realistic lengths.
    
    Args:
        notes: List of (pitch, start_time, duration) tuples
        sequence_length: Length of sequences to create
        
    Returns:
        Tuple of (input_sequences, output_notes, duration_data)
    """
    if len(notes) < sequence_length + 1:
        raise ValueError(f"Not enough notes ({len(notes)}) to create sequences of length {sequence_length}")
    
    pitches = [note[0] for note in notes]
    durations = [note[2] for note in notes]  # PRESERVE DURATION INFORMATION
    
    network_input = []
    network_output = []
    
    # Create overlapping sequences for training
    # Each sequence of length N predicts the next note
    for i in range(0, len(pitches) - sequence_length):
        sequence_in = pitches[i:i + sequence_length]
        sequence_out = pitches[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)
    
    print(f"Created {len(network_input)} training sequences from {len(notes)} notes")
    
    # Reshape for LSTM input format
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    
    # NORMALIZE: Convert MIDI pitches (0-127) to values between 0 and 1
    # This helps with neural network training stability
    network_input = network_input / 128.0
    
    # ONE-HOT ENCODE: Convert output pitches to categorical format
    # This allows the network to predict probability distributions over all possible pitches
    network_output = tf_keras.utils.to_categorical(network_output, 128)
    
    # DEBUG: Print data shapes for verification
    print(f"Input shape: {network_input.shape}, Output shape: {network_output.shape}")
    print(f"Duration data preserved: {len(durations)} values, "
          f"range: {min(durations):.3f}-{max(durations):.3f}s")
    
    return network_input, network_output, durations


def create_model(sequence_length: int = 50) -> tf_keras.Sequential:
    """
    Create an LSTM model for MIDI generation.
    
    ARCHITECTURE NOTES:
    - 3 LSTM layers with 256 units each for learning complex musical patterns
    - Dropout layers (0.3) prevent overfitting
    - Final dense layer outputs probability distribution over 128 MIDI pitches
    
    Args:
        sequence_length: Length of input sequences
        
    Returns:
        Compiled tf_keras model
    """
    model = tf_keras.Sequential()
    
    # FIRST LSTM LAYER: Learn basic sequential patterns
    # return_sequences=True passes outputs to next LSTM layer
    model.add(layers.LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(layers.Dropout(0.3))  # Prevent overfitting
    
    # SECOND LSTM LAYER: Learn higher-level musical structures
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.3))
    
    # THIRD LSTM LAYER: Final sequence processing
    # return_sequences=False outputs only the final state
    model.add(layers.LSTM(256))
    model.add(layers.Dropout(0.3))
    
    # OUTPUT LAYER: Softmax activation gives probability distribution over all 128 MIDI pitches
    model.add(layers.Dense(128, activation='softmax'))
    
    # COMPILE: Categorical crossentropy is standard for multi-class classification
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    # DEBUG: Print model summary
    print("Model architecture:")
    model.summary()
    
    return model


def train_model(
    midi_files: List[str], 
    sequence_length: int = 50, 
    epochs: int = 50, 
    batch_size: int = 64
) -> Tuple[tf_keras.Sequential, List[Tuple[int, float, float]], List[float]]:
    """
    Train the model on a list of MIDI files.
    
    TRAINING PROCESS:
    1. Load and extract notes from all MIDI files
    2. Create training sequences preserving duration information
    3. Train LSTM model to predict next note in sequence
    
    Args:
        midi_files: List of paths to MIDI files
        sequence_length: Length of sequences for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple of (trained_model, extracted_notes, duration_data)
    """
    all_notes = []
    successful_files = 0
    
    print(f"Processing {len(midi_files)} MIDI files...")
    
    # LOAD AND EXTRACT: Gather notes from all training files
    for i, file in enumerate(midi_files):
        print(f"Processing file {i+1}/{len(midi_files)}: {os.path.basename(file)}")
        midi_data = load_midi_file(file)
        if midi_data:
            notes = extract_notes(midi_data)
            if notes:  # Only add if we successfully extracted notes
                all_notes.extend(notes)
                successful_files += 1
            else:
                print(f"WARNING: No notes extracted from {file}")
        else:
            print(f"ERROR: Failed to load {file}")
    
    if not all_notes:
        raise ValueError("No notes were successfully extracted from any MIDI files!")
    
    print(f"\nSuccessfully processed {successful_files}/{len(midi_files)} files")
    print(f"Total notes extracted: {len(all_notes)}")
    
    # CREATE TRAINING DATA: Convert notes to sequences
    network_input, network_output, duration_data = create_sequences_with_duration(all_notes, sequence_length)
    
    # RESHAPE FOR LSTM: [samples, time steps, features]
    # LSTM expects 3D input: (batch_size, sequence_length, input_features)
    network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))
    
    print(f"Final training data shape: {network_input.shape}")
    print(f"Starting training for {epochs} epochs with batch size {batch_size}")
    
    # CREATE AND TRAIN MODEL
    model = create_model(sequence_length)
    
    # TRAINING: Fit model to the data
    # The model learns to predict the next note given a sequence of previous notes
    history = model.fit(
        network_input, 
        network_output, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.2,  # Use 20% of data for validation
        verbose=1
    )
    
    # SAVE TRAINING HISTORY for analysis
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nTraining completed. Final loss: {final_loss:.4f}, Final validation loss: {final_val_loss:.4f}")
    
    if final_val_loss > final_loss * 2:
        print("WARNING: Validation loss much higher than training loss - model may be overfitting!")
    
    return model, all_notes, duration_data


def generate_notes_with_duration(
    model: tf_keras.Sequential, 
    seed_notes: List[int], 
    duration_data: List[float],
    num_notes: int = 100, 
    temperature: float = 1.0
) -> List[Tuple[int, float]]:
    """
    Generate a sequence of notes with realistic durations using the trained model.
    
    GENERATION PROCESS:
    1. Start with seed sequence
    2. Use model to predict next note
    3. Sample from prediction with temperature control
    4. Assign realistic duration based on training data
    5. Repeat for desired number of notes
    
    Args:
        model: Trained tf_keras model
        seed_notes: Starting sequence of notes
        duration_data: Duration information from training data
        num_notes: Number of notes to generate
        temperature: Randomness factor (higher = more random)
        
    Returns:
        List of (pitch, duration) tuples
    """
    print(f"Generating {num_notes} notes with temperature {temperature}")
    
    # VALIDATE INPUT
    if len(seed_notes) != model.input_shape[1]:
        raise ValueError(f"Seed sequence length ({len(seed_notes)}) must match model input shape ({model.input_shape[1]})")
    
    # PREPARE DURATION STATISTICS
    # We'll sample durations from the training data to maintain realistic note lengths
    duration_weights = np.ones(len(duration_data))  # Equal probability for all durations
    
    print(f"Using duration data: {len(duration_data)} samples, "
          f"range: {min(duration_data):.3f}-{max(duration_data):.3f}s")
    
    # Start with the seed sequence
    pattern = seed_notes.copy()
    prediction_output = []

    # GENERATE NOTES ONE BY ONE
    for i in range(num_notes):
        if i % 20 == 0:  # Progress indicator
            print(f"Generated {i}/{num_notes} notes...")
            
        # PREPARE INPUT: Reshape and normalize for model prediction
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / 128.0  # Normalize to match training data
        
        # GET PREDICTION: Model outputs probability distribution over all pitches
        prediction = model.predict(x, verbose=0)
        
        # APPLY TEMPERATURE: Control randomness in generation
        # Lower temperature = more predictable, higher temperature = more random
        if temperature != 1.0:
            prediction = np.log(prediction + 1e-8) / temperature  # Add small epsilon to prevent log(0)
            prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        # SAMPLE NOTE: Choose pitch based on predicted probabilities
        try:
            index = random.choices(range(128), weights=prediction[0], k=1)[0]
        except Exception as e:
            print(f"WARNING: Error in sampling, using argmax. Error: {e}")
            index = np.argmax(prediction[0])
        
        # ASSIGN DURATION: Sample realistic duration from training data
        try:
            duration = random.choices(duration_data, weights=list(duration_weights), k=1)[0]
        except Exception as e:
            print(f"WARNING: Error sampling duration, using mean. Error: {e}")
            duration = np.mean(duration_data)
        
        # ADD TO OUTPUT
        prediction_output.append((index, duration))
        
        # UPDATE PATTERN: Slide window for next prediction
        pattern = np.append(pattern[1:], index)
    
    print(f"Generation complete! Generated {len(prediction_output)} notes")
    
    # DEBUG: Print statistics about generated notes
    gen_pitches = [note[0] for note in prediction_output]
    gen_durations = [note[1] for note in prediction_output]
    print(f"Generated pitch range: {min(gen_pitches)}-{max(gen_pitches)}")
    print(f"Generated duration range: {min(gen_durations):.3f}-{max(gen_durations):.3f}s")
    print(f"Average generated duration: {np.mean(gen_durations):.3f}s")
    
    return prediction_output


def create_midi_from_notes_with_duration(note_sequence: List[Tuple[int, float]], tempo: int = 120) -> pretty_midi.PrettyMIDI:
    """
    Create a MIDI file from a sequence of notes with specified durations.
    
    IMPROVEMENT: This version uses the actual durations from the generated notes
    instead of fixed short durations, resulting in more realistic music.
    
    Args:
        note_sequence: List of (pitch, duration) tuples
        tempo: Tempo in beats per minute
        
    Returns:
        PrettyMIDI object with the generated notes
    """
    print(f"Creating MIDI with {len(note_sequence)} notes at {tempo} BPM")
    
    # CREATE MIDI OBJECT
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # CREATE INSTRUMENT (Piano)
    instrument = pretty_midi.Instrument(program=0)  # Program 0 = Acoustic Grand Piano
    
    # ADD NOTES WITH REALISTIC DURATIONS
    current_time = 0.0
    for i, (pitch, duration) in enumerate(note_sequence):
        # VALIDATE NOTE PARAMETERS
        if pitch < 0 or pitch > 127:
            print(f"WARNING: Invalid pitch {pitch} at position {i}, clamping to valid range")
            pitch = max(0, min(127, pitch))
            
        if duration <= 0:
            print(f"WARNING: Invalid duration {duration} at position {i}, using default")
            duration = 0.5  # Default half-second duration
        
        # CREATE NOTE with realistic duration
        note = pretty_midi.Note(
            velocity=80,  # Volume (0-127), 80 is moderately loud
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration
        )
        instrument.notes.append(note)
        
        # ADVANCE TIME: Notes play sequentially (could be modified for polyphony)
        current_time += duration
    
    # ADD INSTRUMENT TO MIDI
    midi.instruments.append(instrument)
    
    print(f"MIDI created: {current_time:.2f} seconds total duration")
    return midi


def midi_generation_pipeline(
    midi_folder: str, 
    output_file: str, 
    sequence_length: int = 50, 
    epochs: int = 50, 
    notes_to_generate: int = 200, 
    temperature: float = 1.0
) -> tf_keras.Sequential:
    """
    Complete pipeline for training on MIDI files and generating new MIDI with realistic durations.
    
    PIPELINE STEPS:
    1. Load all MIDI files from folder
    2. Extract notes with duration information
    3. Train LSTM model on note sequences
    4. Generate new notes with realistic durations
    5. Create and save output MIDI file
    
    Args:
        midi_folder: Path to folder containing MIDI files
        output_file: Path where the generated MIDI will be saved
        sequence_length: Length of sequences for training and generation
        epochs: Number of training epochs
        notes_to_generate: Number of notes to generate
        temperature: Randomness factor for generation
        
    Returns:
        Trained tf_keras model
    """
    print("=" * 60)
    print("MIDI GENERATION PIPELINE WITH REALISTIC DURATIONS")
    print("=" * 60)
    
    # FIND MIDI FILES
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))
    
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_folder}")
    
    print(f"Found {len(midi_files)} MIDI files in {midi_folder}")
    
    # TRAIN MODEL
    print("\n" + "=" * 30)
    print("TRAINING PHASE")
    print("=" * 30)
    model, all_notes, duration_data = train_model(midi_files, sequence_length, epochs)
    
    # PREPARE SEED SEQUENCE
    print("\n" + "=" * 30)
    print("GENERATION PHASE")
    print("=" * 30)
    
    # Get seed sequence from the original data
    pitches = [note[0] for note in all_notes]
    if len(pitches) < sequence_length:
        raise ValueError(f"Not enough notes in training data ({len(pitches)}) for sequence length {sequence_length}")
    
    start_index = random.randint(0, len(pitches) - sequence_length)
    seed_notes = pitches[start_index:start_index + sequence_length]
    
    print(f"Using seed sequence starting at index {start_index}")
    print(f"Seed notes: {seed_notes[:10]}..." + (f" (and {len(seed_notes)-10} more)" if len(seed_notes) > 10 else ""))
    
    # GENERATE NEW NOTES with realistic durations
    generated_notes = generate_notes_with_duration(
        model, seed_notes, duration_data, notes_to_generate, temperature
    )
    
    # CREATE OUTPUT MIDI
    print("\n" + "=" * 30)
    print("MIDI CREATION PHASE")
    print("=" * 30)
    
    midi_output = create_midi_from_notes_with_duration(generated_notes)
    
    # SAVE OUTPUT
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    midi_output.write(output_file)
    
    print(f"\n{'='*60}")
    print(f"SUCCESS! Generated MIDI saved to: {output_file}")
    print(f"Total duration: {midi_output.get_end_time():.2f} seconds")
    print(f"Notes generated: {len(generated_notes)}")
    print(f"{'='*60}")
    
    return model


# Example usage with comprehensive error handling
if __name__ == "__main__":
    try:
        midi_folder = "data/midi/testing/Cymatics Nebula MIDI Collection/Pop"
        output_file = "data/midi/testing/Cymatics Nebula MIDI Collection/temp/tensoroutput.mid"
        
        # VERIFY PATHS
        if not os.path.exists(midi_folder):
            print(f"ERROR: MIDI folder does not exist: {midi_folder}")
            exit(1)
        
        # RUN PIPELINE
        model = midi_generation_pipeline(
            midi_folder=midi_folder,
            output_file=output_file,
            sequence_length=50,
            epochs=10,  # Reduced for testing - increase for better results
            notes_to_generate=200,
            temperature=1.2  # Slightly random for more interesting results
        )
        
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)