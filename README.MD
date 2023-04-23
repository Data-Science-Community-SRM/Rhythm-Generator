 ![Data Science Community](https://github.com/Data-Science-Community-SRM/template/raw/master/Header.png?raw=true)
# Sound Spirit

Effortlessly generate original melodies and harmonies, tailored to your preferences. Say goodbye to creative blocks, and say hello to a new era of music creation with SoundSpirit.

## Dataset
ComMU has 11,144 MIDI samples that consist of short note sequences created by professional composers with their corresponding 12 metadata.
They consist of short note sequences(4,8,16 bar), and are organized into 12 different metadata. they are as follows: BPM, Genre, Key, Track-instrument, Track-role, Time signature, Pitch range, Number of Measures, Chord progression, Min Velocity, Max Velocity, Rhythm.

## Data Processing
1. The dataset is read and the midi files are converted into a sequence of integers representing musical notes and durations.  
2. The sequences are then divided into input output pairs where the input is a sequence of notes and the output is the next note in the sequence. 
3. The input-output pairs are split into smaller segments (windows) of fixed size. These windows are then used to create training examples by using the elements of each window as input and the following element as output. This allows the model to learn the relationship between the input and output elements and predict the output, given a new input. 
4. The training examples are then split into batches and shuffled before feeding them into the model for training. 
5. During inference, the model generates new sequences of notes by iteratively predicting the next note based on the current sequence of notes.

## Functionalities
* The ability to create original musical compositions, including melodies, harmonies and rhythms
* The  application will be able to generate musical variations or improvisations based on the user's given theme or motiff
* User will be able to interact with the application in real-time such as providing input or making adjustments during the music generation process
* User will be able to save the generated music

## Setup
* Clone this repository
* Install required packages

```pip install -r requirements.txt```
    
## Instructions to Run
* Download the file mentioned below and place it in the 'REMI-tempo-chord-checkpoint' folder inside the 'files' folder  - 
       
   [Model.data](https://drive.google.com/file/d/152BHtFlyN69q53r4GFVlEP1y-jFyhvL0/view?usp=sharing)
* After downloading the required files and installing the necessary packages, run the Flask server

```python app.py```

* After the midi file has been generated use any of the following links to convert it to an mp3 file - 
   
   [Converter 1](https://www.zamzar.com/convert/midi-to-mp3/)
   
   [Converter 2](https://audio.online-convert.com/convert/midi-to-mp3)
   
   [Converter 3](https://converter.app/midi-to-mp3/)



     
     
