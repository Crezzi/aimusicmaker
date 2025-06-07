import glob
import os
import random
from collections import defaultdict
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pretty_midi
import tf_keras
from tf_keras import layers


def load_midi_file(file_path: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Load a MIDI file and return a PrettyMIDI object.
    Returns None if the file can't be loaded.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(file_path)
        print(
            f"Loaded {file_path}: {len(midi_data.instruments)} instruments, "
            f"{midi_data.get_end_time():.2f}s duration"
        )
        return midi_data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_polyphonic_events_with_duration(
    midi_data: pretty_midi.PrettyMIDI, instrument_index: int = 0, time_step: float = 0.125
) -> List[Tuple[List[int], float]]:
    """
    Enhanced version that better captures note durations and musical structure.
    """
    if len(midi_data.instruments) == 0:
        print("WARNING: No instruments found in MIDI file")
        return []

    if instrument_index >= len(midi_data.instruments):
        print(f"WARNING: Instrument index {instrument_index} out of range. Using instrument 0.")
        instrument_index = 0

    instrument = midi_data.instruments[instrument_index]

    # Create events based on note onsets and offsets for better duration modeling
    events = []
    note_events = []

    # Collect all note start/end events
    for note in instrument.notes:
        note_events.append(("start", note.start, note.pitch))
        note_events.append(("end", note.end, note.pitch))

    # Sort by time
    note_events.sort(key=lambda x: x[1])

    # Track active notes and create events at each change
    active_notes = set()
    last_time = 0

    for event_type, time, pitch in note_events:
        if time > last_time and active_notes:
            # Create event for current state
            duration = time - last_time
            if duration >= time_step * 0.5:  # Only include events with meaningful duration
                events.append((sorted(list(active_notes)), duration))

        if event_type == "start":
            active_notes.add(pitch)
        else:
            active_notes.discard(pitch)

        last_time = time

    # Add final event if there are still active notes
    if active_notes:
        events.append((sorted(list(active_notes)), time_step))

    print(f"Extracted {len(events)} events with natural durations")
    return events


def quantize_duration(duration: float, time_step: float = 0.125) -> float:
    """
    Quantize duration to musical subdivisions.
    """
    # Common musical durations (in terms of time_step units)
    musical_durations = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]  # 1/16 to whole notes

    # Convert to time_step units
    duration_units = duration / time_step

    # Find closest musical duration
    closest = min(musical_durations, key=lambda x: abs(x - duration_units))

    return closest * time_step


def encode_polyphonic_events_with_duration(
    events: List[Tuple[List[int], float]], max_notes: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced encoding that includes duration information.

    Returns:
        Tuple of (note_encodings, duration_encodings)
    """
    note_encodings = []
    duration_encodings = []

    # Create duration categories for classification
    duration_categories = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]  # Common durations

    for pitches, duration in events:
        if len(pitches) > max_notes:
            continue

        # Note encoding (same as before)
        note_encoding = np.zeros(128, dtype=np.float32)
        for pitch in pitches:
            if 0 <= pitch <= 127:
                note_encoding[pitch] = 1.0

        # Duration encoding - find closest category
        quantized_duration = quantize_duration(duration)
        duration_category = min(
            range(len(duration_categories)), key=lambda i: abs(duration_categories[i] - quantized_duration)
        )

        note_encodings.append(note_encoding)
        duration_encodings.append(duration_category)

    return np.array(note_encodings), np.array(duration_encodings)


def create_enhanced_polyphonic_model(sequence_length: int = 20) -> tf_keras.Model:
    """
    Create an enhanced model that predicts both notes and durations.
    """
    # Input layer
    input_notes = layers.Input(shape=(sequence_length, 128), name="note_input")
    input_durations = layers.Input(shape=(sequence_length,), name="duration_input")

    # Duration embedding
    duration_embedding = layers.Embedding(input_dim=8, output_dim=16)(input_durations)
    duration_embedding = layers.Reshape((sequence_length, 16))(duration_embedding)

    # Combine note and duration information
    combined = layers.Concatenate()([input_notes, duration_embedding])

    # LSTM layers
    lstm1 = layers.LSTM(512, return_sequences=True)(combined)
    lstm1 = layers.Dropout(0.3)(lstm1)

    lstm2 = layers.LSTM(512, return_sequences=True)(lstm1)
    lstm2 = layers.Dropout(0.3)(lstm2)

    lstm3 = layers.LSTM(512)(lstm2)
    lstm3 = layers.Dropout(0.3)(lstm3)

    # Dense layer for feature extraction
    dense = layers.Dense(256, activation="relu")(lstm3)
    dense = layers.Dropout(0.3)(dense)

    # Output heads
    note_output = layers.Dense(128, activation="sigmoid", name="note_output")(dense)
    duration_output = layers.Dense(8, activation="softmax", name="duration_output")(dense)

    model = tf_keras.Model(inputs=[input_notes, input_durations], outputs=[note_output, duration_output])

    model.compile(
        optimizer="adam",
        loss={"note_output": "binary_crossentropy", "duration_output": "sparse_categorical_crossentropy"},
        metrics={"note_output": "binary_accuracy", "duration_output": "accuracy"},
    )

    print("Enhanced polyphonic model architecture:")
    model.summary()

    return model


def create_enhanced_sequences(
    note_encodings: np.ndarray, duration_encodings: np.ndarray, sequence_length: int = 20
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create training sequences with both note and duration information.
    """
    if len(note_encodings) < sequence_length + 1:
        raise ValueError(f"Not enough events ({len(note_encodings)}) to create sequences")

    input_notes = []
    input_durations = []
    output_notes = []
    output_durations = []

    for i in range(len(note_encodings) - sequence_length):
        # Input sequences
        note_seq = note_encodings[i : i + sequence_length]
        duration_seq = duration_encodings[i : i + sequence_length]

        # Output (next event)
        next_note = note_encodings[i + sequence_length]
        next_duration = duration_encodings[i + sequence_length]

        input_notes.append(note_seq)
        input_durations.append(duration_seq)
        output_notes.append(next_note)
        output_durations.append(next_duration)

    return ([np.array(input_notes), np.array(input_durations)], [np.array(output_notes), np.array(output_durations)])


def generate_enhanced_sequence(
    model: tf_keras.Model,
    seed_notes: np.ndarray,
    seed_durations: np.ndarray,
    num_events: int = 50,
    temperature: float = 1.0,
    note_threshold: float = 0.3,
) -> List[Tuple[List[int], float]]:
    """
    Enhanced generation with better musical variety.
    """
    duration_categories = [0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0]

    pattern_notes = seed_notes.copy()
    pattern_durations = seed_durations.copy()
    generated_events = []

    print(f"Generating {num_events} enhanced events...")

    for i in range(num_events):
        # Prepare inputs
        x_notes = np.reshape(pattern_notes, (1, pattern_notes.shape[0], pattern_notes.shape[1]))
        x_durations = np.reshape(pattern_durations, (1, pattern_durations.shape[0]))

        # Get predictions
        note_pred, duration_pred = model.predict([x_notes, x_durations], verbose=0)
        note_pred = note_pred[0]
        duration_pred = duration_pred[0]

        # Apply temperature to note predictions
        if temperature != 1.0 and temperature > 0:
            note_pred = np.clip(note_pred, 1e-8, 1 - 1e-8)
            logits = np.log(note_pred / (1 - note_pred))
            logits = logits / temperature
            note_pred = 1 / (1 + np.exp(-logits))

        # Generate notes with musical intelligence
        active_notes = generate_musical_notes(note_pred, note_threshold, i)

        # Sample duration
        duration_idx = np.random.choice(len(duration_pred), p=duration_pred)
        duration = duration_categories[duration_idx]

        generated_events.append((active_notes, duration))

        # Update patterns
        new_note_encoding = np.zeros(128, dtype=np.float32)
        for pitch in active_notes:
            if 0 <= pitch < 128:
                new_note_encoding[pitch] = 1.0

        pattern_notes = np.vstack([pattern_notes[1:], new_note_encoding])
        pattern_durations = np.append(pattern_durations[1:], duration_idx)

    return generated_events


def generate_musical_notes(note_pred: np.ndarray, base_threshold: float, event_idx: int) -> List[int]:
    """
    Generate notes with musical intelligence to create variety in chords and single notes.
    """
    # Dynamic threshold based on musical context
    if event_idx % 8 < 2:  # Downbeats - favor chords
        threshold = base_threshold * 0.7
        max_notes = 4
        min_notes = 2
    elif event_idx % 4 == 0:  # Strong beats - mix of chords and single notes
        threshold = base_threshold * 0.8
        max_notes = 3
        min_notes = 1
    else:  # Weak beats - favor single notes
        threshold = base_threshold * 1.2
        max_notes = 2
        min_notes = 0

    # Get candidate notes
    candidates = []
    for pitch, prob in enumerate(note_pred):
        if prob > threshold:
            candidates.append((pitch, prob))

    # Sort by probability
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Select notes based on musical rules
    active_notes = []

    if len(candidates) >= min_notes:
        # Take the most probable notes up to max_notes
        for pitch, prob in candidates[:max_notes]:
            active_notes.append(pitch)
    elif len(candidates) > 0:
        # Take what we have
        for pitch, prob in candidates:
            active_notes.append(pitch)
    else:
        # Force selection if no candidates
        if np.max(note_pred) > 0.01:
            best_notes = np.argsort(note_pred)[-min(2, max_notes) :]
            for pitch in best_notes:
                if note_pred[pitch] > 0.01:
                    active_notes.append(int(pitch))

    # Musical post-processing
    if len(active_notes) > 1:
        active_notes = filter_musical_intervals(active_notes)

    return sorted(active_notes)


def filter_musical_intervals(notes: List[int]) -> List[int]:
    """
    Filter notes to create more musical intervals and chords.
    """
    if len(notes) <= 1:
        return notes

    notes = sorted(notes)
    filtered = [notes[0]]  # Always keep the bass note

    for note in notes[1:]:
        interval = note - filtered[-1]

        # Prefer musical intervals (avoid harsh dissonances)
        if interval >= 3:  # At least a minor third
            # Check for good intervals: 3,4,5,7,8,9,12 semitones
            if interval in [3, 4, 5, 7, 8, 9, 12] or interval >= 12:
                filtered.append(note)
            elif len(filtered) == 1:  # If only bass note, be more lenient
                filtered.append(note)

    return filtered


def train_enhanced_model(
    midi_files: List[str], sequence_length: int = 20, epochs: int = 50, batch_size: int = 32
) -> Tuple[tf_keras.Model, np.ndarray, np.ndarray]:
    """
    Train the enhanced model.
    """
    all_events = []
    successful_files = 0

    print(f"Processing {len(midi_files)} MIDI files for enhanced training...")

    for i, file in enumerate(midi_files):
        print(f"Processing file {i+1}/{len(midi_files)}: {os.path.basename(file)}")
        midi_data = load_midi_file(file)
        if midi_data:
            events = extract_polyphonic_events_with_duration(midi_data)
            if events:
                all_events.extend(events)
                successful_files += 1

    if not all_events:
        raise ValueError("No events were successfully extracted!")

    print(f"\nSuccessfully processed {successful_files}/{len(midi_files)} files")
    print(f"Total events extracted: {len(all_events)}")

    # Encode events
    note_encodings, duration_encodings = encode_polyphonic_events_with_duration(all_events)

    # Create sequences
    X, y = create_enhanced_sequences(note_encodings, duration_encodings, sequence_length)

    print(f"Training data shapes: Notes {X[0].shape}, Durations {X[1].shape}")

    # Create and train model
    model = create_enhanced_polyphonic_model(sequence_length)

    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1)

    print("Enhanced training completed!")
    return model, note_encodings, duration_encodings


def enhanced_midi_pipeline(
    midi_folder: str,
    output_file: str,
    sequence_length: int = 20,
    epochs: int = 50,
    events_to_generate: int = 100,
    temperature: float = 1.0,
    note_threshold: float = 0.4,
) -> tf_keras.Model:
    """
    Complete enhanced pipeline for varied polyphonic MIDI generation.
    """
    print("=" * 60)
    print("ENHANCED POLYPHONIC MIDI GENERATION PIPELINE")
    print("=" * 60)

    midi_files = glob.glob(os.path.join(midi_folder, "*.mid")) + glob.glob(os.path.join(midi_folder, "*.midi"))

    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_folder}")

    print(f"Found {len(midi_files)} MIDI files")

    # Train model
    model, note_encodings, duration_encodings = train_enhanced_model(midi_files, sequence_length, epochs)

    # Generate music
    print("\n" + "GENERATION PHASE".center(60, "="))

    start_idx = random.randint(0, len(note_encodings) - sequence_length)
    seed_notes = note_encodings[start_idx : start_idx + sequence_length]
    seed_durations = duration_encodings[start_idx : start_idx + sequence_length]

    generated_events = generate_enhanced_sequence(
        model, seed_notes, seed_durations, events_to_generate, temperature, note_threshold
    )

    # Create MIDI
    midi_output = create_midi_from_polyphonic_events(generated_events, tempo=120)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    midi_output.write(output_file)

    # Print statistics
    chords = [e for e in generated_events if len(e[0]) > 1]
    singles = [e for e in generated_events if len(e[0]) == 1]
    rests = [e for e in generated_events if len(e[0]) == 0]

    print(f"\nGeneration Statistics:")
    print(f"Chords: {len(chords)} ({len(chords)/len(generated_events)*100:.1f}%)")
    print(f"Single notes: {len(singles)} ({len(singles)/len(generated_events)*100:.1f}%)")
    print(f"Rests: {len(rests)} ({len(rests)/len(generated_events)*100:.1f}%)")

    if chords:
        chord_sizes = [len(e[0]) for e in chords]
        print(f"Chord sizes: {min(chord_sizes)}-{max(chord_sizes)} notes")

    durations = [e[1] for e in generated_events]
    print(f"Duration range: {min(durations):.3f}s - {max(durations):.3f}s")

    print(f"\nSUCCESS! Enhanced MIDI saved to: {output_file}")

    return model


def create_midi_from_polyphonic_events(
    events: List[Tuple[List[int], float]], tempo: int = 120
) -> pretty_midi.PrettyMIDI:
    """
    Create MIDI from polyphonic events (same as original but with better logging).
    """
    print(f"Creating MIDI with {len(events)} events at {tempo} BPM")

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)

    current_time = 0.0
    total_notes = 0

    for pitches, duration in events:
        for pitch in pitches:
            if 0 <= pitch <= 127:
                note = pretty_midi.Note(velocity=80, pitch=int(pitch), start=current_time, end=current_time + duration)
                instrument.notes.append(note)
                total_notes += 1

        current_time += duration

    midi.instruments.append(instrument)
    print(f"Created MIDI: {current_time:.2f}s duration, {total_notes} notes")

    return midi


# Example usage
if __name__ == "__main__":
    try:
        midi_folder = "data/midi/testing/Cymatics Nebula MIDI Collection/EDM"
        output_file = "data/midi/testing/Cymatics Nebula MIDI Collection/temp/enhanced_polyphonic_output.mid"

        if not os.path.exists(midi_folder):
            print(f"ERROR: MIDI folder does not exist: {midi_folder}")
            exit(1)

        model = enhanced_midi_pipeline(
            midi_folder=midi_folder,
            output_file=output_file,
            sequence_length=20,
            epochs=5,
            events_to_generate=150,
            temperature=0.8,
            note_threshold=0.3,  # Lower threshold for more note variety
        )

        print("\nEnhanced polyphonic pipeline completed successfully!")

    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
