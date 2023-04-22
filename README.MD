# Sound Spirit

Effortlessly generate original melodies and harmonies, tailored to your preferences. Say goodbye to creative blocks, and say hello to a new era of music creation with SoundSpirit.

## Dataset
ComMU has 11,144 MIDI samples that consist of short note sequences created by professional composers with their corresponding 12 metadata.
They consist of short note sequences(4,8,16 bar), and are organized into 12 different metadata. they are as follows: BPM, Genre, Key, Track-instrument, Track-role, Time signature, Pitch range, Number of Measures, Chord progression, Min Velocity, Max Velocity, Rhythm.

## Data Processing
* Import necessary libraries for data manipulation and Processing
* Load the data from a CSV file to a pandas DataFrame
* Define constants such as MAJOR_KEY, MINOR_KEY, KEY_MAP, TIME_SIG_MAP, PITCH_RANGE_MAP and map their data values to their corresponding numeric representations.
* Map columns in the DataFrame to their numeric representations using the defined constants.
* Perform one-hot encoding on selected columns in the DataFrame for categorical data.
* Scale numeric columns in the DataFrame to a common scale between 0 and 1.
* Update the DataFrame with the mapped, encoded, and scaled values for further analysis or machine learning tasks.

## Functionalities
* The ability to create original musical compositions, including melodies, harmonies and rhythms
* The  application will be able to generate musical variations or improvisations based on the user's given theme or motiff
* User will be able to interact with the application in real-time such as providing input or making adjustments during the music generation process
* User will be able to save the generated music

## Setup
1. Clone this repository
2. Install required packages

```pip install -r requirements.txt```
    
## Instructions to Run
* After installing the required packages run the flask server

```python app.py```

* After the midi file has been generated use any of the following links to convert it to an mp3 file - 
   
   [Converter 1](https://www.zamzar.com/convert/midi-to-mp3/)
   
   [Converter 2](https://audio.online-convert.com/convert/midi-to-mp3)
   
   [Converter 3](https://converter.app/midi-to-mp3/)



     
     
