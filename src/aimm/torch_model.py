import os
import pickle
import random
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

#################################################
# Part 1: MIDI Processing with PrettyMIDI
#################################################

class MidiProcessor:
    def __init__(self, sequence_length=32, step=1):
        """
        Initialize the MIDI processor
        
        Args:
            sequence_length: Length of sequences for training
            step: Step size for creating sequences
        """
        self.sequence_length = sequence_length
        self.step = step
        self.notes = []
        self.note_to_int = {}
        self.int_to_note = {}
        self.vocab_size = 0
        
    def extract_notes_from_midi(self, midi_file_path):
        """
        Extract notes from a MIDI file using PrettyMIDI
        
        Args:
            midi_file_path: Path to the MIDI file
            
        Returns:
            List of extracted notes/chords
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            notes = []
            
            # Go through all instruments
            for instrument in midi_data.instruments:
                # Skip drum tracks
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Encode note as: pitch_start-time_duration
                        # This format captures more musical information
                        encoded_note = f"{note.pitch}_{round(note.start, 2)}_{round(note.end - note.start, 2)}"
                        notes.append(encoded_note)
            
            # Sort notes by start time
            notes.sort(key=lambda x: float(x.split('_')[1]))
            return notes
            
        except Exception as e:
            print(f"Error processing {midi_file_path}: {e}")
            return []
    
    def extract_notes_simplified(self, midi_file_path):
        """
        Extract just the pitch values and durations from a MIDI file
        (Simpler alternative to extract_notes_from_midi)
        """
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file_path)
            notes = []
            
            # Go through all instruments
            for instrument in midi_data.instruments:
                # Skip drum tracks
                if not instrument.is_drum:
                    for note in instrument.notes:
                        # Just encode pitch and duration
                        encoded_note = f"{note.pitch}_{round(note.end - note.start, 2)}"
                        notes.append(encoded_note)
            
            return notes
            
        except Exception as e:
            print(f"Error processing {midi_file_path}: {e}")
            return []
            
    def process_midi_directory(self, directory_path):
        """
        Process all MIDI files in a directory
        
        Args:
            directory_path: Path to directory containing MIDI files
        """
        all_notes = []
        
        # Find all MIDI files
        midi_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                     if f.endswith('.mid') or f.endswith('.midi')]
        
        print(f"Found {len(midi_files)} MIDI files")
        
        # Process each file
        for file_path in tqdm(midi_files):
            notes = self.extract_notes_from_midi(file_path)
            if notes:
                all_notes.extend(notes)
                
        print(f"Extracted {len(all_notes)} notes from all files")
        self.notes = all_notes
        
        # Create mappings
        unique_notes = sorted(set(all_notes))
        self.vocab_size = len(unique_notes)
        
        print(f"Number of unique notes/events: {self.vocab_size}")
        
        # Create dictionaries to map between notes and integers
        self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note = {i: note for i, note in enumerate(unique_notes)}
        
        return all_notes
    
    def prepare_sequences(self):
        """
        Prepare input and output sequences for training
        
        Returns:
            X: Input sequences
            y: Target sequences
        """
        # Create input sequences and corresponding outputs
        network_input = []
        network_output = []
        
        # Create sequences
        for i in range(0, len(self.notes) - self.sequence_length, self.step):
            sequence_in = self.notes[i:i + self.sequence_length]
            sequence_out = self.notes[i + self.sequence_length]
            
            # Convert sequence to integers
            network_input.append([self.note_to_int[note] for note in sequence_in])
            network_output.append(self.note_to_int[sequence_out])
            
        # Reshape and normalize input
        n_patterns = len(network_input)
        network_input = np.reshape(network_input, (n_patterns, self.sequence_length))
        
        # One-hot encode the output
        network_output = to_categorical(network_output, num_classes=self.vocab_size)
        
        return network_input, network_output
    
    def save_mappings(self, filename="note_mappings.pkl"):
        """Save the note mappings to a file"""
        data = {
            "note_to_int": self.note_to_int,
            "int_to_note": self.int_to_note,
            "vocab_size": self.vocab_size
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
    def load_mappings(self, filename="note_mappings.pkl"):
        """Load the note mappings from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        self.note_to_int = data["note_to_int"]
        self.int_to_note = data["int_to_note"]
        self.vocab_size = data["vocab_size"]
        
    def generate_midi_from_sequence(self, note_sequence, output_file='generated_output.mid'):
        """
        Convert a sequence of note representations back to a MIDI file
        
        Args:
            note_sequence: List of encoded notes
            output_file: Output MIDI file path
        """
        # Create a PrettyMIDI object
        midi = pretty_midi.PrettyMIDI()
        
        # Create an instrument (piano)
        instrument = pretty_midi.Instrument(program=0)  # 0 is piano
        
        current_time = 0.0
        
        # Add all notes
        for encoded_note in note_sequence:
            parts = encoded_note.split('_')
            
            if len(parts) == 3:  # Full format: pitch_start-time_duration
                pitch = int(parts[0])
                start_time = float(parts[1])
                duration = float(parts[2])
                
                note = pretty_midi.Note(
                    velocity=100,  # Default velocity
                    pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                current_time += duration
                
            elif len(parts) == 2:  # Simplified format: pitch_duration
                pitch = int(parts[0])
                duration = float(parts[1])
                
                note = pretty_midi.Note(
                    velocity=100,  # Default velocity
                    pitch=pitch,
                    start=current_time,
                    end=current_time + duration
                )
                instrument.notes.append(note)
                current_time += duration
        
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
        
        # Write the MIDI file
        midi.write(output_file)
        print(f"Generated MIDI file saved to {output_file}")

#################################################
# Part 2: PyTorch LSTM/RNN Model for MIDI Generation
#################################################

# Dataset class for MIDI data
class MidiDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.LongTensor(inputs)
        self.targets = torch.LongTensor(targets)
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# PyTorch LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Embed the input
        embedded = self.embedding(x)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Apply dropout to the final LSTM output
        lstm_out = self.dropout(lstm_out[:, -1, :])
        
        # Pass through the final layer
        output = self.fc(lstm_out)
        
        return output

class LSTMMidiGenerator:
    def __init__(self, processor, model_path=None):
        """
        Initialize the LSTM/RNN model for MIDI generation
        
        Args:
            processor: MidiProcessor instance
            model_path: Path to a pre-trained model (optional)
        """
        self.processor = processor
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self, hidden_size=256, num_layers=3):
        """
        Build the LSTM model
        
        Args:
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM layers
        """
        self.model = LSTMModel(
            input_size=self.processor.vocab_size,
            hidden_size=hidden_size,
            output_size=self.processor.vocab_size,
            num_layers=num_layers,
            dropout=0.3
        ).to(self.device)
        
        return self.model
    
    def build_simple_model(self, hidden_size=128):
        """
        Build a simpler LSTM model for quicker training
        
        Args:
            hidden_size: Size of hidden layer
        """
        self.model = LSTMModel(
            input_size=self.processor.vocab_size,
            hidden_size=hidden_size,
            output_size=self.processor.vocab_size,
            num_layers=1,
            dropout=0.2
        ).to(self.device)
        
        return self.model
    
    def prepare_data(self, network_input, network_output, batch_size=64):
        """Prepare data for PyTorch training"""
        # Convert one-hot output to class indices
        if len(network_output.shape) > 1 and network_output.shape[1] > 1:
            network_output = np.argmax(network_output, axis=1)
        
        # Create dataset and dataloader
        dataset = MidiDataset(network_input, network_output)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )
        
        return dataloader
    
    def train(self, network_input, network_output, epochs=50, batch_size=64, checkpoint_path="model_best.pt"):
        """
        Train the LSTM model
        
        Args:
            network_input: Input sequences
            network_output: Target sequences
            epochs: Number of training epochs
            batch_size: Batch size for training
            checkpoint_path: Path to save model checkpoints
        """
        if not self.model:
            raise ValueError("Model has not been built")
        
        # Convert one-hot output to class indices if needed
        if len(network_output.shape) > 1 and network_output.shape[1] > 1:
            network_output = np.argmax(network_output, axis=1)
        
        # Create dataloader
        dataloader = self.prepare_data(network_input, network_output, batch_size)
        
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_loss = float('inf')
        history = {'loss': []}
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            # Progress bar
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for inputs, targets in progress_bar:
                # Move data to device
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update total loss
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            # Calculate average loss for the epoch
            avg_loss = total_loss / len(dataloader)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Model saved to {checkpoint_path}")
                
            # Early stopping (simple implementation)
            if epoch > 10 and all(history['loss'][i-5] < history['loss'][i] for i in range(epoch-4, epoch+1)):
                print("Early stopping triggered")
                break
        
        # Load the best model
        self.model.load_state_dict(torch.load(checkpoint_path))
        
        return history
    
    def generate_sequence(self, seed_sequence, num_notes_to_generate=100, temperature=1.0):
        """
        Generate a sequence of notes
        
        Args:
            seed_sequence: Initial sequence to seed the generation
            num_notes_to_generate: Number of notes to generate
            temperature: Controls randomness (higher = more random)
            
        Returns:
            List of generated notes
        """
        if not self.model:
            raise ValueError("Model has not been built or loaded")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Convert seed sequence to integer representation
        pattern = [self.processor.note_to_int[note] for note in seed_sequence]
        prediction_output = []
        
        # Generate notes
        with torch.no_grad():
            for _ in range(num_notes_to_generate):
                # Convert pattern to tensor
                x = torch.LongTensor([pattern[-self.processor.sequence_length:]]).to(self.device)
                
                # Make prediction
                output = self.model(x)
                
                # Convert to probabilities
                probabilities = torch.softmax(output / temperature, dim=1).cpu().numpy()[0]
                
                # Sample from the distribution
                index = self.sample_with_temperature(probabilities, temperature)
                
                # Get the note corresponding to the index
                result = self.processor.int_to_note[index]
                prediction_output.append(result)
                
                # Update pattern
                pattern.append(index)
                
        return prediction_output
    
    def sample_with_temperature(self, probabilities, temperature):
        """
        Sample an index from a probability array with temperature
        
        Args:
            probabilities: Array of probabilities
            temperature: Temperature parameter
            
        Returns:
            Sampled index
        """
        if temperature == 0:
            # If temperature is 0, return the most probable index
            return np.argmax(probabilities)
        
        # Sample from the distribution
        choices = range(len(probabilities))
        return np.random.choice(choices, p=probabilities)
    
    def save_model(self, model_path="lstm_model.pt"):
        """Save the model to a file"""
        if self.model:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'vocab_size': self.processor.vocab_size,
                'sequence_length': self.processor.sequence_length,
            }, model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save")
    
    def load_model(self, model_path="lstm_model.pt", hidden_size=256, num_layers=3):
        """Load the model from a file"""
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # If it's just the state dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # Create the model
                self.build_model(hidden_size, num_layers)
                
                # Load the state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Create the model
                self.build_model(hidden_size, num_layers)
                
                # Load the state dict
                self.model.load_state_dict(checkpoint)
                
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file {model_path} does not exist")

#################################################
# Part 3: Markov Model for MIDI Generation
#################################################

class MarkovMidiGenerator:
    def __init__(self, order=2):
        """
        Initialize the Markov model for MIDI generation
        
        Args:
            order: Order of the Markov model (how many previous notes to consider)
        """
        self.order = order
        self.transitions = defaultdict(Counter)
        self.start_tokens = []
    
    def train(self, notes):
        """
        Train the Markov model on a sequence of notes
        
        Args:
            notes: List of notes
        """
        # Store potential starting sequences
        for i in range(len(notes) - self.order):
            self.start_tokens.append(tuple(notes[i:i+self.order]))
        
        # Build transition probabilities
        for i in range(len(notes) - self.order):
            # Current sequence of notes
            current = tuple(notes[i:i+self.order])
            
            # Next note
            next_note = notes[i + self.order]
            
            # Update transition counter
            self.transitions[current][next_note] += 1
    
    def generate_sequence(self, seed=None, num_notes=100):
        """
        Generate a sequence of notes using the Markov model
        
        Args:
            seed: Optional starting sequence (if None, one will be chosen randomly)
            num_notes: Number of notes to generate
            
        Returns:
            List of generated notes
        """
        result = []
        
        # Choose a starting sequence
        if seed and len(seed) >= self.order:
            current = tuple(seed[-self.order:])
        else:
            current = random.choice(self.start_tokens)
            
        # Add the starting sequence to the result
        result.extend(current)
        
        # Generate the rest of the notes
        for _ in range(num_notes - self.order):
            if current in self.transitions and self.transitions[current]:
                # Get the possible next notes and their counts
                next_notes = self.transitions[current]
                
                # Choose the next note weighted by frequency
                choices, weights = zip(*next_notes.items())
                next_note = random.choices(choices, weights=weights, k=1)[0]
                
                # Add the next note to the result
                result.append(next_note)
                
                # Update the current sequence
                current = tuple(result[-self.order:])
            else:
                # If we reach a dead end, choose a new random starting point
                new_start = random.choice(self.start_tokens)
                result.append(new_start[-1])
                current = tuple(result[-self.order:])
        
        return result
    
    def save_model(self, filename="markov_model.pkl"):
        """Save the model to a file"""
        data = {
            "order": self.order,
            "transitions": dict(self.transitions),
            "start_tokens": self.start_tokens
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
            
    def load_model(self, filename="markov_model.pkl"):
        """Load the model from a file"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        self.order = data["order"]
        self.transitions = defaultdict(Counter)
        
        # Convert back from regular dict to defaultdict(Counter)
        for k, v in data["transitions"].items():
            self.transitions[k] = Counter(v)
            
        self.start_tokens = data["start_tokens"]

#################################################
# Part 4: Usage Example
#################################################

def lstm_training_pipeline(midi_dir, output_dir="output", sequence_length=32):
    """Complete pipeline for training an LSTM model"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and process MIDI files
    processor = MidiProcessor(sequence_length=sequence_length)
    processor.process_midi_directory(midi_dir)
    
    # Save mappings for later use
    processor.save_mappings(os.path.join(output_dir, "note_mappings.pkl"))
    
    # Prepare sequences for training
    network_input, network_output = processor.prepare_sequences()
    
    # Initialize and build the model
    generator = LSTMMidiGenerator(processor)
    generator.build_model()
    
    # Train the model
    checkpoint_path = os.path.join(output_dir, "lstm_weights.best.pt")
    history = generator.train(
        network_input, 
        network_output, 
        epochs=50,
        batch_size=64,
        checkpoint_path=checkpoint_path
    )
    
    # Save the complete model
    generator.save_model(os.path.join(output_dir, "lstm_model.pt"))
    
    # Plot training history if matplotlib is available
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'])
        plt.title('Model Loss During Training')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(output_dir, "training_loss.png"))
    except:
        print("Matplotlib not available, skipping loss plot")
    
    # Generate a sample
    seed_sequence = processor.notes[:sequence_length]
    generated_notes = generator.generate_sequence(
        seed_sequence, 
        num_notes_to_generate=100
    )
    
    # Create a MIDI file from the generated notes
    processor.generate_midi_from_sequence(
        generated_notes,
        output_file=os.path.join(output_dir, "lstm_generated_output.mid")
    )
    
    return generator

def markov_training_pipeline(midi_dir, output_dir="output", markov_order=2):
    """Complete pipeline for training a Markov model"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and process MIDI files
    processor = MidiProcessor()
    notes = processor.process_midi_directory(midi_dir)
    
    # Initialize and train the Markov model
    markov_generator = MarkovMidiGenerator(order=markov_order)
    markov_generator.train(notes)
    
    # Save the model
    markov_generator.save_model(os.path.join(output_dir, "markov_model.pkl"))
    
    # Generate a sample
    seed = notes[:markov_order] if notes else None
    generated_notes = markov_generator.generate_sequence(
        seed=seed,
        num_notes=100
    )
    
    # Create a MIDI file from the generated notes
    processor.generate_midi_from_sequence(
        generated_notes,
        output_file=os.path.join(output_dir, "markov_generated_output.mid")
    )
    
    return markov_generator

if __name__ == "__main__":
    # Example usage
    MIDI_DIR = "path/to/your/midi/files"
    OUTPUT_DIR = "generated_output"
    
    # Choose which model to train (or both)
    train_lstm = True
    train_markov = True
    
    if train_lstm:
        print("Training LSTM model...")
        lstm_generator = lstm_training_pipeline(MIDI_DIR, OUTPUT_DIR)
        
    if train_markov:
        print("Training Markov model...")
        markov_generator = markov_training_pipeline(MIDI_DIR, OUTPUT_DIR)