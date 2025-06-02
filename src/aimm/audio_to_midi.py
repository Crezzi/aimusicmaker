import os
from music21 import converter, key as music21key
from datetime import datetime, timedelta
from pathlib import Path
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

def transpose_to_c_major(input_path, output_path):
    # Load the MIDI file using music21 to estimate key
    midi_stream = converter.parse(input_path)
    
    # Analyze key
    k = midi_stream.analyze('key')
    print(f"Estimated key: {k}")
    
    # Calculate the interval needed to transpose to C major or A minor
    if k.mode == "major":
        interval = music21key.Key("C").tonic.pitchClass - k.tonic.pitchClass
    else:
        interval = music21key.Key("A").tonic.pitchClass - k.tonic.pitchClass

    # Transpose the stream
    transposed_stream = midi_stream.transpose(interval)

    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write out the transposed MIDI file
    transposed_stream.write("midi", output_path)
    print(f"Saved transposed file to {output_path}")

if __name__ == "__main__":
    #print(main())
    output = '''
    data/midi/testing/Cymatics Nebula MIDI Collection/Pop_transposed/Cymatics - Wake Up MIDI - G Maj.mid
    '''
    input2 = output.replace("Pop_transposed","Pop")
    output = output.strip()
    input2 = input2.strip()
    transpose_to_c_major(input2,output)