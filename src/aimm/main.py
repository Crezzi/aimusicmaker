from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
from pretty_midi import PrettyMIDI


class Main:
    def __init__(self) -> None: ...

    @staticmethod
    def convert_audio_to_midi(audio_path: str) -> PrettyMIDI:
        output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
        return midi_data

    def main(self) -> None: ...


if __name__ == "__main__":
    ...
