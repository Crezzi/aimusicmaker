# AI Midi Maker
---

## About the Project
---

AI Midi Maker is just that! A program to help make midi files with the use of machine learning algorithms.

We are still in the process of developing.

Contributors:
- Mihir Amin
- Niall Danahey
- Jessica Stainton-Simmons

## Checklist
---

### Front End

1. Start with a text-based interface for the user.
    - Input for:
        - Artist
        - Folder
        - Album
    - Output for:
        - A file in a specified location
2. Move to a visual UI for users to input a folder/artist name/album.
    - Input for:
        - Artist
        - Folder
        - Album
    - Output for:
        - A file in a specified location
        - A midi display
3. Make it look nice,

### Back End

1.  Make a program that an input from an album or artist, and gets the songs (the audio files) from that artist/album.
2.  Seperate those songs into acapellas and instrumentals. The acapells will not be used
3. Turn those instrumentals into `.mid` files.
4. Shorten the `.mid` files so that they are all the same length.
5. Normalize those `.mid` files to all be in the same key.
6. Split the `.mid` files into bass, drum, and melody sections.
7. Convert the `.mid` files into `Midi` objects.
8. Use a machine learning algorithm to train a model.
    - Markov Chain Model
    - Bayesian Predictor Model
    - Recursion Learning
    - Supervised Learning
9. Train the model.