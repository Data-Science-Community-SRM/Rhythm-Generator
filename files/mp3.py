from midi2audio import FluidSynth
import os
import fnmatch

# Set path to your SoundFont file here
sf2path = 'soundfont.sf2'

fs = FluidSynth(sf2path)

for root, dirnames, filenames in os.walk('result/from_scratch'):
    for filename in fnmatch.filter(filenames, '*.midi'):
        midi_path = os.path.join(root, filename)
        audio_path = os.path.join('result_mp3', os.path.splitext(filename)[0] + '.wav')
        fs.midi_to_audio(midi_path, audio_path)