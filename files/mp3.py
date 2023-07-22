
import pyaudio  
import wave    
import pygame   
import fnmatch  
import os      

mp3_output_dir = 'result_mp3'


do_ffmpeg_convert = True   
do_wav_cleanup = True      
sample_rate = 44100        
channels = 2                
buffer = 1024              
mp3_bitrate = 128          
input_device = 1           



def play_music(music_file):

    try:
        pygame.mixer.music.load(music_file)
        
    except pygame.error:
        print ("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
        return
        
    pygame.mixer.music.play()

bitsize = -16  
pygame.mixer.init(sample_rate, bitsize, channels, buffer)
pygame.mixer.music.set_volume(0)
format = pyaudio.paInt16
audio = pyaudio.PyAudio()



try:

    matches = []
    for root, dirnames, filenames in os.walk('result/from_scratch'):
        for filename in fnmatch.filter(filenames, '*.midi'):
            matches.append(os.path.join(root, filename))
            

    for song in matches:


        file_name = os.path.splitext(os.path.basename(song))[0]
        new_file = file_name + '.wav'


        stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, input_device_index=input_device, frames_per_buffer=buffer)
        

        print("Playing " + file_name + ".mid\n")
        play_music(song)
        
        frames = []
        

        while pygame.mixer.music.get_busy():
            frames.append(stream.read(buffer))
            
        stream.stop_stream()
        stream.close()
        wave_file = wave.open(new_file, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(sample_rate)
        
        print("Saving " + new_file)   

        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        if do_ffmpeg_convert:
            os.system('ffmpeg -i ' + new_file + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + str(channels) + ' -ar ' + str(sample_rate) + ' -vn ' + os.path.join(mp3_output_dir, file_name + '.mp3'))
            
            if do_wav_cleanup:        
                os.remove(new_file)
         
    audio.terminate()    
 
except KeyboardInterrupt:
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit