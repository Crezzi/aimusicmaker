from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from pretty_midi import PrettyMIDI
from audio_to_midi import transpose_to_c_major
from music21 import key
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

class Main:
    def __init__(self) -> None: ...

    @staticmethod
    def convert_audio_to_midi(audio_path: str) -> PrettyMIDI:
        output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
        return midi_data

    def main(self) -> None: ...


if __name__ == "__main__":
    transpose_to_c_major("data/midi/testing/Cymatics Nebula MIDI Collection/Pop/Cymatics - Love MIDI - F Maj.mid","data/audio/data/output.md/training_and_testing_data/output.mid")
