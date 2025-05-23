import mido
from mido import MidiFile, MidiTrack, Message

def change_note_in_midi(input_file, output_file, original_note, new_note):
    mid = MidiFile(input_file)
    new_mid = MidiFile()

    for i, track in enumerate(mid.tracks):
        new_track = MidiTrack()
        new_mid.tracks.append(new_track)
        for msg in track:
            if msg.type in ['note_on', 'note_off'] and msg.note == original_note:
                msg.note = new_note
            new_track.append(msg)

    new_mid.save(output_file)
    print(f"Saved modified MIDI to {output_file}")

# Example usage
change_note_in_midi("input.mid", "output.mid", original_note=60, new_note=62)
