from model import RythmTransformer
from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

def main():
    # declare model
    model = RythmTransformer(
        checkpoint='REMI-tempo-chord-checkpoint',
        is_training=True)
    # prepare data
    midi_paths = glob('dataset/raw/*.mid') 
    training_data = model.prepare_data(midi_paths=midi_paths)

    output_checkpoint_folder = 'REMI-finetune' 
    if not os.path.exists(output_checkpoint_folder):
        os.mkdir(output_checkpoint_folder)
    
    # finetune
    model.finetune(
        training_data=training_data,
        output_checkpoint_folder=output_checkpoint_folder)
    # close
    model.close()

if __name__ == '__main__':
    main()
