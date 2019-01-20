from __future__ import print_function, division
from pydub import AudioSegment

import librosa
import thinkdsp
import thinkplot
import thinkstats2

import numpy as np
import pandas as pd
import os

import warnings
warnings.filterwarnings("ignore")

#globals
harmonics = [1,1/2,1/3,1/4]
precision = 0.01
test_folder_path = "permutations"

def assess_quality(path_to_wav_file):
    f = thinkdsp.read_wave(path_to_wav_file)
    f.normalize()
    spectrum = f.make_spectrum()
    peaks = (spectrum.peaks())
    quality = 1
    i = 1
    count = 1
    while (i<10):
        quality += harmonicNess(peaks[i], peaks[i-1])
        i = i+1
    
    return quality/count

def folderToArray(path_to_folder):
    playlist = []
    for song in os.listdir(path_to_folder):
        if (song.endswith(".wav")):
            playlist.append(path_to_folder + "/" + song)
    return playlist
        
def harmonicNess(ratio_tpl_2, ratio_tpl_1):
    num = 0
    for h in harmonics:
        if (closeEnough(ratio_tpl_2[1]/ratio_tpl_1[1], h, precision)):
            num += (ratio_tpl_2[1]/ratio_tpl_1[1])
    return num
    
def closeEnough(test, value, threshold):   
    if (abs(test-value) < threshold):
        return True
    return False

def getTempoAndBeats(path_to_wav_file):
    y, sr = librosa.load(path_to_wav_file)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo,beats

def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

#does not change pitch
def stretch(path_to_wav_file,new_file_path,old_tempo, new_tempo):
    y, sr = librosa.load(path_to_wav_file)
    stretch = new_tempo/old_tempo
    y_new = librosa.effects.time_stretch(y, stretch)
    #restoring pitch
    y_new = librosa.effects.pitch_shift(y_new, sr, n_steps=1/stretch)

    librosa.output.write_wav(new_file_path, y_new, sr, False)

#start2 is in time
def add(path_song1, path_song2, start2, output_path):
    sound1 = AudioSegment.from_file(path_song1)
    sound2 = AudioSegment.from_file(path_song2)
    sound3 = sound2.overlay(sound1, position=start2)
    sound3.export(output_path, format="wav")

def find_best_overlay(path_song1, path_song2, dir):
    y1, sr1 = librosa.load(path_song1)
    y2, sr2 = librosa.load(path_song2)
    tempo1, beats1 = getTempoAndBeats(path_song1)
    tempo2, beats2 = getTempoAndBeats(path_song2)

    # stretch/shrink tempo
    stretch(path_song2, path_song2, tempo2,tempo1)
    tempo2, beats2 = getTempoAndBeats(path_song2)

    # Second song, by definition, is song with later occurrence of first beat
    ### Only modifying the second song

    results = []
    if (beats1[0] < beats2[0]): 
        for t in range(0, 3):
            # Case t + 1: when the first beat of the first song aligns with the (t + 1)th beat of the second song
            start2 = (beats2[t] - beats1[0])*sr1
            start2 = librosa.core.get_duration(y=y1, sr=start2)
            print("start2: " + str(start2))

            output_path = dir + '/' + path_song1.split('/')[-1] + '_' + path_song2.split('/')[-1] + '_' + str(t) + '.wav'
            add(path_song1, path_song2, start2, output_path)

            quality = assess_quality(output_path)
            results.append((output_path, quality))
    
    return results

# sr1 should equal sr2
if __name__ == "__main__":
    norm = False
    if norm:
        for file in os.listdir(test_folder_path + "/"):
            if (file.endswith(".wav")):
                sound = AudioSegment.from_file(test_folder_path + "/" + file, "mp3")
                normalized_sound = match_target_amplitude(sound, -20.0)
                normalized_sound.export(test_folder_path + "/" + file, format="mp3")

    playlist = folderToArray(test_folder_path)
    permutations = {}
    for i in playlist:
        for j in playlist:
            if (i != j):
                print('Currently PyMashing: ' + i + ' and ' + j)

                permutations[i.split('/')[-1] + '|' + j.split('/')[-1]] = find_best_overlay(i, j, test_folder_path)
                
    with open('permutations/quality.txt', 'w') as f:
        f.write(permutations)