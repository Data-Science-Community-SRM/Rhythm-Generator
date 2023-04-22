import mido
from pydub import AudioSegment, generators

# Load the MIDI file
mid = mido.MidiFile('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/0.midi')

# Set the sample rate and channels for the output WAV file
sample_rate = 44100
channels = 2

# Create an empty audio segment
audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)

# Iterate through each MIDI track and add it to the audio segment
for track in mid.tracks:
    track_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    for message in track:
        if message.type == 'note_on':
            # Map the MIDI note value to a frequency
            frequency = 2 ** ((message.note - 69) / 12) * 440
            # Calculate the duration of the note in milliseconds
            duration = int(message.time * 1000)
            # Generate a sine wave for the note
            note_audio = generators.Sine(frequency).to_audio_segment(duration=duration, volume=-10)
            # Add the note to the track audio
            track_audio += note_audio
    # Add the track audio to the final audio segment
    audio += track_audio

# Export the audio segment to a WAV file
audio.export('output_file.wav', format='wav')

