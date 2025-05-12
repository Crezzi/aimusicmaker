import os
from datetime import datetime, timedelta

# from basic_pitch.train import main HOLY FUCK NO WAY THEY HAVE WHAT WE NEED
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict


def audio_to_midi(audio_path: str, i: int):
    start = datetime.now()
    output, midi_data, note_events = predict(audio_path, ICASSP_2022_MODEL_PATH)
    midi_data.write(f"data/audio/temp/{audio_path[-5]}.mid")
    end = datetime.now()
    print(str(i) + ":", end - start)
    return end - start


def main() -> tuple[timedelta, timedelta, timedelta, timedelta]:
    t1 = audio_to_midi("data/audio/temp/1.mp3", 1)
    t2 = audio_to_midi("data/audio/temp/2.mp3", 2)
    t3 = audio_to_midi("data/audio/temp/3.mp3", 3)
    t4 = audio_to_midi("data/audio/temp/4.mp3", 4)
    return t1, t2, t3, t4


def delete_temp_midis() -> None:
    for i in range(1, 5):
        os.remove(f"data/audio/temp/{i}.mid")


if __name__ == "__main__":
    print(main())
    # delete_temp_midis()
