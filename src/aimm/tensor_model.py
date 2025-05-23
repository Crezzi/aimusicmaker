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
        return midi_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_notes(midi_data: pretty_midi.PrettyMIDI, instrument_index: int = 0) -> List[Tuple[int, float, float]]:
    """
    Extract notes from the first instrument of a PrettyMIDI object.
    
    Args:
        midi_data: PrettyMIDI object
        instrument_index: Index of the instrument to extract notes from
        
    Returns:
        List of (pitch, start_time, duration) tuples
    """
    if len(midi_data.instruments) == 0:
        return []
    
    instrument = midi_data.instruments[instrument_index]
    notes = []
    
    for note in instrument.notes:
        notes.append((note.pitch, note.start, note.end - note.start))
    
    return sorted(notes, key=lambda x: x[1])  # Sort by start time


def create_sequences(notes: List[Tuple[int, float, float]], sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences of notes for training.
    
    Args:
        notes: List of (pitch, start_time, duration) tuples
        sequence_length: Length of sequences to create
        
    Returns:
        Tuple of (input_sequences, output_notes)
    """
    pitches = [note[0] for note in notes]
    
    network_input = []
    network_output = []
    
    # Create sequences
    for i in range(0, len(pitches) - sequence_length):
        sequence_in = pitches[i:i + sequence_length]
        sequence_out = pitches[i + sequence_length]
        network_input.append(sequence_in)
        network_output.append(sequence_out)
    
    # Reshape and normalize
    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length))
    
    # Normalize to values between 0 and 1
    network_input = network_input / 128.0
    
    # One-hot encode outputs
    network_output = tf_keras.utils.to_categorical(network_output, 128)
    
    return network_input, network_output


def create_model(sequence_length: int = 50) -> tf_keras.Sequential:
    """
    Create an LSTM model for MIDI generation.
    
    Args:
        sequence_length: Length of input sequences
        
    Returns:
        Compiled tf_keras model
    """
    model = tf_keras.Sequential()
    
    # Add LSTM layers
    model.add(layers.LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256))
    model.add(layers.Dropout(0.3))
    
    # Output layer
    model.add(layers.Dense(128, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model


def train_model(
    midi_files: List[str], 
    sequence_length: int = 50, 
    epochs: int = 50, 
    batch_size: int = 64
) -> Tuple[tf_keras.Sequential, List[Tuple[int, float, float]]]:
    """
    Train the model on a list of MIDI files.
    
    Args:
        midi_files: List of paths to MIDI files
        sequence_length: Length of sequences for training
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Tuple of (trained_model, extracted_notes)
    """
    all_notes = []
    
    # Load and extract notes from all files
    for file in midi_files:
        midi_data = load_midi_file(file)
        if midi_data:
            notes = extract_notes(midi_data)
            all_notes.extend(notes)
    
    # Process data for training
    network_input, network_output = create_sequences(all_notes, sequence_length)
    
    # Reshape input for LSTM [samples, time steps, features]
    network_input = np.reshape(network_input, (network_input.shape[0], network_input.shape[1], 1))
    
    # Create and train model
    model = create_model(sequence_length)
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size)
    
    return model, all_notes


def generate_notes(
    model: tf_keras.Sequential, 
    seed_notes: List[int], 
    num_notes: int = 100, 
    temperature: float = 1.0
) -> List[int]:
    """
    Generate a sequence of notes using the trained model.
    
    Args:
        model: Trained tf_keras model
        seed_notes: Starting sequence of notes
        num_notes: Number of notes to generate
        temperature: Randomness factor (higher = more random)
        
    Returns:
        List of generated note pitches
    """
    # Start with the seed sequence
    pattern = seed_notes
    prediction_output = []

    # Generate notes
    for _ in range(num_notes):
        # Reshape the input
        x = np.reshape(pattern, (1, len(pattern), 1))
        x = x / 128.0  # Normalize
        
        # Generate prediction
        prediction = model.predict(x)
        
        # Apply temperature for randomness
        if temperature != 1.0:
            prediction = np.log(prediction) / temperature
            prediction = np.exp(prediction) / np.sum(np.exp(prediction))
        
        # Sample from the prediction
        index = random.choices(range(128), weights=prediction[0], k=1)[0]
        
        # Add prediction to output
        prediction_output.append(index)
        
        # Remove first element and append prediction for next iteration
        pattern = np.append(pattern[1:], [index])
    
    return prediction_output


def create_midi_from_notes(note_sequence: List[int], tempo: int = 120) -> pretty_midi.PrettyMIDI:
    """
    Create a MIDI file from a sequence of notes.
    
    Args:
        note_sequence: List of MIDI note pitches
        tempo: Tempo in beats per minute
        
    Returns:
        PrettyMIDI object with the generated notes
    """
    # Create PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    
    # Create instrument
    instrument = pretty_midi.Instrument(program=0)  # Piano
    
    # Set the timing (quarter note length in seconds)
    quarter_note_duration = 60.0 / tempo
    note_duration = quarter_note_duration / 2  # Eighth notes
    
    # Add notes to the instrument
    current_time = 0.0
    for pitch in note_sequence:
        note = pretty_midi.Note(
            velocity=100,  # Velocity (volume)
            pitch=pitch,
            start=current_time,
            end=current_time + note_duration
        )
        instrument.notes.append(note)
        current_time += note_duration
    
    # Add instrument to MIDI object
    midi.instruments.append(instrument)
    
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
    Complete pipeline for training on MIDI files and generating new MIDI.
    
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
    # Get all MIDI files in the folder
    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))
    
    if not midi_files:
        print(f"No MIDI files found in {midi_folder}")
        quit()
    
    print(f"Found {len(midi_files)} MIDI files. Training model...")
    
    # Train model
    model, all_notes = train_model(midi_files, sequence_length, epochs)
    
    # Get seed sequence from the original data
    pitches = [note[0] for note in all_notes]
    start_index = random.randint(0, len(pitches) - sequence_length - 1)
    seed_notes = pitches[start_index:start_index + sequence_length]
    
    print("Generating new notes...")
    # Generate new notes
    generated_notes = generate_notes(model, seed_notes, notes_to_generate, temperature)
    
    # Create MIDI file
    midi_output = create_midi_from_notes(generated_notes)
    
    # Save the generated MIDI
    midi_output.write(output_file)
    print(f"Generated MIDI saved to {output_file}")
    
    return model


# Example usage
if __name__ == "__main__":
    midi_folder = "data/midi/testing/Cymatics Nebula MIDI Collection/Pop"
    output_file = "data/midi/testing/Cymatics Nebula MIDI Collection/temp/tensoroutput.mid"
    
    # Run the pipeline
    model = midi_generation_pipeline(
        midi_folder=midi_folder,
        output_file=output_file,
        sequence_length=50,
        epochs=10,  # Use fewer epochs for testing
        notes_to_generate=200,
        temperature=1.2  # Higher temperature = more randomness
    )