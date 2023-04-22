from flask import Flask,render_template,redirect,request,send_file
import numpy as np
#from midiutil import MIDIFile

from model import RythmTransformer
import os
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main(num_samples, num_bars, temperature, prompt_dir ,output_dir):
    # declare model
    model = RythmTransformer(
        checkpoint='files/REMI-tempo-chord-checkpoint',
        is_training=False)

    from_scratch_path = output_dir + 'from_scratch/'
    with_prompt_path = output_dir + 'with_prompt/'
    
    if not os.path.exists(from_scratch_path):
        os.makedirs(from_scratch_path)
    if not os.path.exists(with_prompt_path):
        os.makedirs(with_prompt_path)

    for i in range(num_samples):
        # generate from scratch
        model.generate(
            n_target_bar=num_bars,
            temperature=temperature,
            output_path=from_scratch_path + '{}.midi'.format(i),
            prompt=None)

        # generate continuation
        #model.generate(
          #  n_target_bar=num_bars,
          #  temperature=temperature,
           # output_path= with_prompt_path + '{}.midi'.format(i),
           # prompt=prompt_dir + f'{i:03}' + '.midi')

    # close model
    model.close()
#myMIDI = MIDIFile(1)
#creating the app
app=Flask(__name__)

@app.route('/',methods=['GET',"POST"])
def index_page():
    return render_template('index.html')

@app.route('/generatemusic')
def rec_page():
    return render_template('generatemusic.html')

@app.route('/generate',methods=['GET','POST'])
def generate():
    if request.method == 'POST':
        f = [x for x in request.form.values()]
        d1 = [np.array(f)]
        print(f)
        num_samples = int(f[0])
        num_bars = int(f[1])
        temperature = float(f[2])
        prompt_dir = './data/evaluation/'
        output_dir = './result/'
        main(num_samples,num_bars,temperature,prompt_dir, output_dir)
        '''new_file = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/0.midi', 'wb')
        myMIDI.writeFile(new_file)
        new_file.close()'''

        '''new_file = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/0.midi', 'rb')
        return send_file(new_file, mimetype='audio/midi')'''
        return render_template('output.html', no_samples = num_samples)

@app.route('/download1')
def sample1():
    new_file = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/0.midi', 'rb')
    return send_file(new_file,mimetype='audio/midi')

@app.route('/download2')
def sample2():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/1.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download3')
def sample3():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/2.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download4')
def sample4():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/3.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download5')
def sample5():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/4.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download6')
def sample6():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/5.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download7')
def sample7():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/6.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download8')
def sample8():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/7.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download9')
def sample9():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/8.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')

@app.route('/download10')
def sample10():
    new_file1 = open('/Users/saisatyajonnalagadda/Documents/satna/Rhythm-Generator/result/from_scratch/9.midi', 'rb')
    return send_file(new_file1,mimetype='audio/midi')


if __name__ == "__main__":
    app.run(debug=True)


'''if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples')
    parser.add_argument("--ns", default= 10, type= int, help= "number of samples")
    parser.add_argument("--nb", default= 16, type = int, help= "number of target bars")
    parser.add_argument("--t", default= 1.2, type = float, help= "temperature")
    parser.add_argument("--p", default= './data/evaluation/', type = str,help= " prompt directory")
    parser.add_argument("--o", default= './result/', type = str, help= "output directory")
    
    args = parser.parse_args()

    num_samples = args.ns
    num_bars = args.nb
    temperature = args.t
    prompt_dir = args.p
    output_dir = args.o


    print("Generating {} samples".format(num_samples))
    print("With {} target bars".format(num_bars))
    print("With {} temperature".format(temperature))
    print("Input promt root folder {}".format(prompt_dir))
    print("Output root folder {}".format(output_dir))
    main(num_samples,num_bars,temperature,prompt_dir, output_dir)'''
