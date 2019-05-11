import scipy.io.wavfile as wav
from threading import Thread
import simpleaudio as sa
import datetime
import multiprocessing
import time
import pyaudio
import wave

import os
import math
from pylab import *

import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output1.wav"
 
p = pyaudio.PyAudio()
 
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

def Record_audio(): 
    
    time.sleep(0.42)
    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    #time.sleep(0.5)
wave1 = sa.WaveObject.from_wave_file("1007.wav")

def record():
    #print(datetime.datetime.now())
    Record_audio()
def play(wave_name):
    
    #print(datetime.datetime.now())
    #print("audio play")
    #time.sleep(0.3)
    #print(datetime.datetime.now())
    wave_file = sa.WaveObject.from_wave_file(wave_name)
    #print(datetime.datetime.now())
    play = wave_file.play()
    time.sleep(1.5)
    #print("audio finish")


def load_labels(filename):
    return [line.rstrip() for line in tf.gfile.FastGFile(filename)]

    
    
  
def load_model(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=None, 
            return_elements=None, 
            name="prefix", 
            op_dict=None, 
            producer_op_list=None
        )
    return graph

 

    
if __name__ == '__main__':
    
    labels_dict = []
    labels_path="ckpts/action_labels1.txt"
    labels_dict = load_labels(labels_path)
    
    graph123 = load_model("ds_ckpts/DSCNN_freeze_model.pb")
    sess2 = tf.Session(graph=graph123)
    input1234 = graph123.get_tensor_by_name('prefix/Placeholder:0')
    output1234 = graph123.get_tensor_by_name('prefix/Softmax:0')
    
    
    #p1 = multiprocessing.Process(name='p1', target=record)
    #p = multiprocessing.Process(name='p', target=play)
    
    
    tot_count = 0
    ans_count = 0
    #time.sleep(1)
    
    for label in labels_dict:
        if label == "yes" and label!="silent" and label!="back_ground":
            print(label)
            
            data_dir = "phy_adv_data/"+label
            wav_files_list =\
            [f for f in os.listdir(data_dir) if f.endswith(".wav")]
            
            for input_file in wav_files_list:
                
                tot_count = tot_count + 1
                #wave1 = sa.WaveObject.from_wave_file(data_dir+'/'+input_file)
                wave_load = data_dir+'/'+input_file
                time.sleep(1)
                print("------------------------------1")
                p1 = multiprocessing.Process(name='p1', target=record)
                p = multiprocessing.Process(name='p', target=play, args=(wave_load,))
                p1.start()
                p.start()
                
                
                p1.join()
                p.join()
                time.sleep(0.5)
                
                r1, d = wav.read("output1.wav")
                d1 = list(d)
                while len(d1)!=16000:
                    d1.append(0)
                d1 = np.array(d1)
                d = d1
                d1 = d1 / 32767
                
                d1 = np.expand_dims(d1, axis=0)
                pred = sess2.run(output1234, feed_dict={input1234:d1})
                pred = np.array(pred)
                print(labels_dict[ np.argmax(pred[0]) ])
                
                print("************************** 2")
                p1 = multiprocessing.Process(name='p1', target=record)
                p = multiprocessing.Process(name='p', target=play, args=(wave_load,))
                p1.start()
                p.start()
                
                
                p1.join()
                p.join()
                time.sleep(0.5)
                
                r1, d = wav.read("output1.wav")
                d1 = list(d)
                while len(d1)!=16000:
                    d1.append(0)
                d1 = np.array(d1)
                d = d1
                d1 = d1 / 32767
                
                d1 = np.expand_dims(d1, axis=0)
                pred = sess2.run(output1234, feed_dict={input1234:d1})
                pred = np.array(pred)
                print(labels_dict[ np.argmax(pred[0]) ])
                if labels_dict[ np.argmax(pred[0]) ] == label:
                    ans_count = ans_count + 1
                #break
    print("ans count: ", ans_count/tot_count)
    
    
