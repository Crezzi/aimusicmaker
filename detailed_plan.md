# AI Music Maker Detailed Plan

## Data Collection

### Task 1: Getting songs from an album/artist
1. **Set up the basic API connections**
   - Create a free Spotify developer account
   - Get API credentials (client ID and client secret)
   - Install the `spotipy` library: `pip install spotipy`

2. **Write a simple script to search for artists**
   - Create a function that takes an artist name and returns their Spotify ID
   - Test with a few well-known artists

3. **Write a function to get albums by an artist**
   - Use the artist ID to retrieve their albums
   - Handle pagination if there are many albums

4. **Write a function to get tracks from an album**
   - Take an album ID and return all tracks in that album
   - Store basic metadata (track name, duration, etc.)

### Task 2: Converting songs to MIDI
1. **Research MIDI sources**
   - Look for existing MIDI datasets of popular songs
   - Consider websites like FreeMidi.org or Hooktheory

2. **Set up a download function**
   - Create a function to download MIDIs based on song names
   - Create a simple caching system to avoid re-downloading

3. **Alternative: Audio-to-MIDI conversion**
   - If direct MIDIs aren't available, install libraries like `librosa` and `mido`
   - Start with extracting notes from simple monophonic audio
   - Test on a single short audio file before scaling up

### Task 3: Creating MIDI objects
1. **Set up a basic MIDI parser**
   - Install the `mido` library: `pip install mido`
   - Create a function that reads a MIDI file and prints basic info

2. **Extract note events**
   - Create a function that extracts note-on and note-off events
   - Store these in a simple list structure

3. **Create a simple MIDI class**
   - Define a basic class to hold MIDI data in memory
   - Include methods to access notes, tempo, and timing

4. **Add metadata storage**
   - Enhance your class to store song title, artist name
   - Create a method to save processed MIDIs to disk

## AI Development

### Task 1: Setting up the AI environment
1. **Install basic ML libraries**
   - Install TensorFlow or PyTorch: `pip install tensorflow` or `pip install torch`
   - Install music-specific libraries: `pip install music21`

2. **Create a simple data loader**
   - Write a function to load MIDI files from a directory
   - Create basic preprocessing to convert MIDI to tensors

3. **Set up a development notebook**
   - Create a Jupyter notebook to experiment with data and models
   - Load a small sample of MIDI files for testing

### Task 2: Creating a basic model
1. **Start with a simple model**
   - Begin with a basic LSTM or RNN architecture
   - Use just 1-2 layers to verify data flow

2. **Define input and output formats**
   - Decide how to represent MIDI events (one-hot encoding, embeddings, etc.)
   - Define sequence length and batch size

3. **Create a training loop**
   - Write a basic function to train for one epoch
   - Add logging to monitor progress

### Task 3: Training and generating output
1. **Implement basic training**
   - Train on a very small dataset first (e.g., 5-10 MIDI files)
   - Verify loss is decreasing and model is learning

2. **Create a simple generation function**
   - Write code to generate new notes from the trained model
   - Start with generating just a few seconds of music

3. **Add temperature/randomness control**
   - Implement a parameter to control randomness in generation
   - Test different values to see how they affect output

4. **Create a MIDI output function**
   - Write code to convert model output back to MIDI format
   - Test by listening to generated output

5. **Integrate with the main system**
   - Create a function that takes input MIDIs and returns generated MIDIs
   - Ensure it can handle the format provided by Niall's component

## Next Steps for the Team

1. **Set up a shared repository**:
   - Create a GitHub repo with a clear folder structure
   - Establish coding conventions and documentation standards
   - Set up basic CI/CD for testing

2. **Define interfaces between components**:
   - Create clear specifications for how the UI will pass data to the collection module
   - Define how the collection module will pass data to the AI module
   - Establish file formats and object structures

3. **Schedule regular check-ins**:
   - Set up weekly meetings to review progress
   - Use issues/tickets to track tasks and blockers