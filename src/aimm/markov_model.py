import os
import pickle
import random
from collections import Counter, defaultdict
from typing import Counter as CounterType
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from pretty_midi import Instrument, Note, PrettyMIDI
from pretty_midi.utilities import note_name_to_number, note_number_to_name


class MarkovMidi:
    """
    An improved Markov model for MIDI generation that considers timing information.
    This version tracks when notes are played relative to each other.
    """

    def __init__(self, order: int = 2) -> None:
        """
        Initialize the Markov model

        Args:
            order: Order of the Markov model (how many previous notes to consider)
        """
        self.order: int = order
        self.transitions: DefaultDict[Tuple[str, ...], CounterType[str]] = defaultdict(Counter)
        self.start_tokens: List[Tuple[str, ...]] = []

        # Track the timing between notes
        self.timing_gaps: DefaultDict[Tuple[str, str], List[float]] = defaultdict(list)

    def extract_notes_from_midi(self, midi_file_path: str) -> List[Tuple[str, float]]:
        """
        Extract notes from a MIDI file with timing information

        Args:
            midi_file_path: Path to the MIDI file

        Returns:
            List of tuples (note_encoding, start_time)
        """
        try:
            midi_data: PrettyMIDI = PrettyMIDI(midi_file_path)
            notes_with_timing: List[Tuple[str, float]] = []

            # Go through all instruments
            for instrument in midi_data.instruments:
                # Skip drum tracks
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Enhanced format: store pitch, duration, velocity, and start time
                        encoded_note: str = f"{note.pitch}_{round(note.end - note.start, 2)}_{note.velocity}"
                        notes_with_timing.append((encoded_note, note.start))

            # Sort notes by their start time
            notes_with_timing.sort(key=lambda x: x[1])
            return notes_with_timing

        except Exception as e:
            print(f"Error processing {midi_file_path}: {e}")
            return []

    def train_on_midi_directory(self, directory_path: str) -> List[Tuple[str, float]]:
        """
        Train the Markov model on all MIDI files in a directory

        Args:
            directory_path: Path to directory containing MIDI files

        Returns:
            List of all notes with timing extracted from the MIDI files
        """
        all_notes_with_timing: List[Tuple[str, float]] = []

        # Find all MIDI files
        midi_files: List[str] = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".mid") or f.endswith(".midi")
        ]

        print(f"Found {len(midi_files)} MIDI files")

        # Process each file
        for file_path in midi_files:
            print(f"Processing {file_path}")
            notes_with_timing: List[Tuple[str, float]] = self.extract_notes_from_midi(file_path)
            if notes_with_timing:
                all_notes_with_timing.extend(notes_with_timing)

        print(f"Extracted {len(all_notes_with_timing)} notes from all files")

        # Train the Markov model
        self.train(all_notes_with_timing)

        return all_notes_with_timing

    def train(self, notes_with_timing: List[Tuple[str, float]]) -> None:
        """
        Train the Markov model on a sequence of notes with timing information

        Args:
            notes_with_timing: List of tuples (note_encoding, start_time)
        """
        if len(notes_with_timing) <= self.order:
            print("Warning: Not enough notes to train properly")
            return

        # Extract just the notes (without timing) for the Markov chain
        notes: List[str] = [note for note, _ in notes_with_timing]

        # Store potential starting sequences
        for i in range(len(notes) - self.order):
            self.start_tokens.append(tuple(notes[i : i + self.order]))

        # Build transition probabilities
        for i in range(len(notes) - self.order):
            # Current sequence of notes
            current: Tuple[str, ...] = tuple(notes[i : i + self.order])

            # Next note
            next_note: str = notes[i + self.order]

            # Update transition counter
            self.transitions[current][next_note] += 1

            # Record timing information between consecutive notes
            if i + self.order < len(notes_with_timing):
                current_note = notes[i + self.order - 1]
                current_time = notes_with_timing[i + self.order - 1][1]
                next_time = notes_with_timing[i + self.order][1]

                # Calculate the time gap between these notes
                time_gap = next_time - current_time

                # Store this timing information
                self.timing_gaps[(current_note, next_note)].append(time_gap)

        print(
            f"Model trained with {len(self.transitions)} unique sequences and {len(self.timing_gaps)} timing relationships"
        )

    def generate_sequence(self, seed: Optional[List[str]] = None, num_notes: int = 100) -> List[Tuple[str, float]]:
        """
        Generate a sequence of notes with timing using the Markov model

        Args:
            seed: Optional starting sequence (if None, one will be chosen randomly)
            num_notes: Number of notes to generate

        Returns:
            List of tuples (note_encoding, start_time)
        """
        if not self.transitions:
            raise ValueError("Model has not been trained")

        result_notes: List[str] = []
        result_with_timing: List[Tuple[str, float]] = []

        # Choose a starting sequence
        if seed and len(seed) >= self.order:
            current: Tuple[str, ...] = tuple(seed[-self.order :])
        else:
            if not self.start_tokens:
                raise ValueError("No start tokens available")
            current = random.choice(self.start_tokens)

        # Add the starting sequence to the result
        result_notes.extend(current)

        # Initialize timing for the starting sequence (arbitrary start at 0.0)
        current_time = 0.0
        for note in current:
            result_with_timing.append((note, current_time))
            # For initial sequence, use small default gaps
            current_time += 0.25

        # Generate the rest of the notes
        for _ in range(num_notes - self.order):
            if current in self.transitions and self.transitions[current]:
                # Get the possible next notes and their counts
                next_notes: CounterType[str] = self.transitions[current]

                # Choose the next note weighted by frequency
                choices, weights = zip(*next_notes.items())
                next_note: str = random.choices(choices, weights=weights, k=1)[0]

                # Determine timing for this note
                prev_note = result_notes[-1]
                timing_key = (prev_note, next_note)

                if timing_key in self.timing_gaps and self.timing_gaps[timing_key]:
                    # Use the learned timing between these specific notes
                    time_gap = random.choice(self.timing_gaps[timing_key])
                else:
                    # If we don't have timing data for this transition, use a reasonable default
                    # Extract duration from previous note as a fallback
                    try:
                        parts = prev_note.split("_")
                        if len(parts) >= 2:
                            prev_duration = float(parts[1])
                            time_gap = prev_duration * 0.8  # Slight overlap by default
                        else:
                            time_gap = 0.25  # Default gap
                    except (ValueError, IndexError):
                        time_gap = 0.25  # Default gap

                # Calculate the new note's start time
                new_time = result_with_timing[-1][1] + time_gap

                # Add the next note to the results
                result_notes.append(next_note)
                result_with_timing.append((next_note, new_time))

                # Update the current sequence
                current = tuple(result_notes[-self.order :])
            else:
                # If we reach a dead end, choose a new random starting point
                if not self.start_tokens:
                    break

                new_start: Tuple[str, ...] = random.choice(self.start_tokens)
                next_note = new_start[-1]

                # Add a slightly larger gap when transitioning to a new sequence
                new_time = result_with_timing[-1][1] + 0.5 if result_with_timing else 0.0

                result_notes.append(next_note)
                result_with_timing.append((next_note, new_time))
                current = tuple(result_notes[-self.order :])

        return result_with_timing

    def generate_midi(self, note_sequence: List[Tuple[str, float]], output_file: str = "markov_output.mid") -> None:
        """
        Convert a sequence of notes with timing to a MIDI file

        Args:
            note_sequence: List of tuples (note_encoding, start_time)
            output_file: Output file path
        """
        # Create a PrettyMIDI object
        midi: PrettyMIDI = PrettyMIDI()

        # Create an instrument (piano)
        instrument: Instrument = Instrument(program=0)  # 0 is piano

        # Add each note to the instrument
        for note_str, start_time in note_sequence:
            parts: List[str] = note_str.split("_")

            if len(parts) >= 2:  # At minimum we need pitch and duration
                pitch: int = int(parts[0])
                duration: float = float(parts[1])

                # Use velocity if available, otherwise default to 100
                velocity: int = int(parts[2]) if len(parts) >= 3 else 100

                note: Note = Note(velocity=velocity, pitch=pitch, start=start_time, end=start_time + duration)

                instrument.notes.append(note)

        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)

        # Write the MIDI file
        midi.write(output_file)
        print(f"Generated MIDI file saved to {output_file}")

    def save_model(self, filename: str = "timing_aware_markov_model.pkl") -> None:
        """Save the model to a file"""
        data: Dict = {
            "order": self.order,
            "transitions": dict(self.transitions),
            "start_tokens": self.start_tokens,
            "timing_gaps": {k: v for k, v in self.timing_gaps.items()},  # Convert to regular dict
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Model saved to {filename}")

    def load_model(self, filename: str = "timing_aware_markov_model.pkl") -> None:
        """Load the model from a file"""
        with open(filename, "rb") as f:
            data: Dict = pickle.load(f)

        self.order = data["order"]
        self.transitions = defaultdict(Counter)

        # Convert back from regular dict to defaultdict(Counter)
        for k, v in data["transitions"].items():
            self.transitions[k] = Counter(v)

        self.start_tokens = data["start_tokens"]

        # Load timing information
        self.timing_gaps = defaultdict(list)
        for k, v in data["timing_gaps"].items():
            self.timing_gaps[k] = v

        print(f"Model loaded from {filename}")


# Example usage
def timing_aware_markov_example(
    midi_directory: str, output_directory: str = "output"
) -> MarkovMidi:
    """
    Example of using the MarkovMidi

    Args:
        midi_directory: Directory containing MIDI files
        output_directory: Directory to save outputs

    Returns:
        Trained Markov model
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Create and train the model
    model: MarkovMidi = MarkovMidi(order=2)
    model.train_on_midi_directory(midi_directory)

    # Save the model
    model.save_model(os.path.join(output_directory, "timing_aware_markov_model.pkl"))

    # Generate a new sequence
    generated_sequence = model.generate_sequence(num_notes=100)

    # Create a MIDI file
    model.generate_midi(
        generated_sequence, output_file=os.path.join(output_directory, "timing_aware_markov_output.mid")
    )

    return model


if __name__ == "__main__":
    # Example usage - uncomment and modify path to use
    # timing_aware_markov_example("path/to/midi/files", "output")
    pass
