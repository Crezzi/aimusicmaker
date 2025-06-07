import glob
import os
from pathlib import Path

import mido
from mido import Message, MidiFile, MidiTrack

from aimm.audio_to_midi import transpose_a_dir, transpose_to_c_major
from aimm.tensor_model import enhanced_midi_pipeline


def change_note_in_midi(input_file, output_file, original_note, new_note):
    mid = MidiFile(input_file)
    new_mid = MidiFile()

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.type in ["note_on", "note_off"] and msg.note == original_note:
                msg.note = new_note
            new_track.append(msg)

    new_mid.save(output_file)
    print(f"Saved modified MIDI to {output_file}")


def transpose():
    transpose_a_dir("uploads", "website_folder/transposed")
    folder_path = "website_folder/download"  # Change this to your folder path

    # Delete all files (but not subfolders or their contents)
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)


def tensor_AI(input_dir: str = "website_folder/transposed", output_file: str = "website_folder/download/AIMM.midi"):
    model = enhanced_midi_pipeline(
        midi_folder=input_dir,
        output_file=output_file,
        sequence_length=20,
        epochs=5,
        events_to_generate=150,
        temperature=0.8,
        note_threshold=0.3,  # Lower threshold for more note variety
    )
    folder_path = "website_folder/transposed"  # Change this to your folder path

    # Delete all files (but not subfolders or their contents)
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        if os.path.isfile(file_path):
            os.remove(file_path)


if __name__ == "__main__":
    tensor_AI()
