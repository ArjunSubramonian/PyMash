from __future__ import print_function, division

from pydub import AudioSegment
import librosa
import thinkdsp
import thinkplot
import thinkstats2

import numpy as np
import pandas as pd
import soundfile as sf
import os
import json

import sys

import warnings
warnings.filterwarnings("ignore")

# global constants
harmonics = [1,1/2,1/3,1/4]
precision = 0.01
test_folder_path = ""

to_wav = False
norm = False
convert_PCM_16 = False
qual = False

def assess_quality(path_to_wav_file):
    f = thinkdsp.read_wave(path_to_wav_file)
    # makes niceness based on the amplitude of the peaks
    f.normalize()
    spectrum = f.make_spectrum()
    peaks = (spectrum.peaks())

    quality = 1
    i = 1
    count = 1
    while (i < 10):
        quality += harmonicness(peaks[i], peaks[i-1])
        i += 1
    
    return quality/count
        
def harmonicness(ratio_tpl_2, ratio_tpl_1):
    num = 0
    for h in harmonics:
        if within_threshold(ratio_tpl_2[1]/ratio_tpl_1[1], h, precision):
            num += (ratio_tpl_2[1]/ratio_tpl_1[1])
    return num
    
def within_threshold(test, value, threshold):   
    if abs(test-value) < threshold:
        return True
    return False

def get_tempo_and_beats(song_data):
    y, sr = song_data
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

# does not change pitch
def stretch(song_data, new_file_path, old_tempo, new_tempo):
    y, sr = song_data
    stretch = new_tempo/old_tempo

    # making sure the song isn't slowed down too much
    ### librosa estimates tempo based on pulses, so a song with
    ### only downbeats would result in a much lower tempo
    while stretch < 1/2:
        stretch = stretch * 2
    while stretch > 2:
        stretch = stretch/2
    y_new = librosa.effects.time_stretch(y, stretch)

    # restoring pitch
    y_new = librosa.effects.pitch_shift(y_new, sr, n_steps=1/stretch)
    librosa.output.write_wav(new_file_path, y_new, sr, False)

# start2 is in time
def add(path_song1, path_song2, start2, output_path):
    sound1 = AudioSegment.from_file(path_song1)
    sound2 = AudioSegment.from_file(path_song2)
    sound3 = sound2.overlay(sound1, position=start2)
    sound3.export(output_path, format="wav")

def find_best_overlay(path_song1, song1_data, path_song2, song2_data, name_song1, name_song2):
    y1, sr1 = song1_data
    y2, sr2 = song2_data
    tempo1, beats1 = get_tempo_and_beats(song1_data)
    tempo2, beats2 = get_tempo_and_beats(song2_data)

    # stretch/shrink tempo
    stretch(song2_data, path_song2, tempo2, tempo1)
    tempo2, beats2 = get_tempo_and_beats(librosa.load(path_song2))

    ### Only modifying the song with later occurrence of first beat
    results = []
    if beats1[0] < beats2[0]: 
        for t in range(0, 3):
            # Case t + 1: when the first beat of the first song aligns with the (t + 1)th beat of the second song
            start2 = (beats2[t] - beats1[0]) * sr1
            start2 = librosa.core.get_duration(y=y1, sr=start2)
            print("start2: " + str(start2))

            output_path = test_folder_path + '/' + name_song1 + '_' + name_song2 + '_' + str(t) + '.wav'
            add(path_song1, path_song2, start2, output_path)

            if qual:
                quality = assess_quality(output_path)
                results.append((output_path, quality))
    else:
        for t in range(0, 3):
            # Case t + 1: when the first beat of the first song aligns with the (t + 1)th beat of the second song
            start1 = (beats1[t] - beats2[0]) * sr2
            start1 = librosa.core.get_duration(y=y2, sr=start1)
            print("start1: " + str(start1))

            output_path = test_folder_path + '/' + name_song2 + '_' + name_song1 + '_' + str(t) + '.wav'
            add(path_song2, path_song1, start1, output_path)

            if qual:
                quality = assess_quality(output_path)
                results.append((output_path, quality))
    
    if qual:
        return results

if __name__ == "__main__":
    
    try:
        test_folder_path = sys.argv[1]
    except:
        print("Please provide music library folder path.")

    try:
        options = [x.upper() for x in sys.argv[2:]]

        if "WAV" in options:
            to_wav = True
        if "NORM" in options:
            norm = True
        if "PCM16" in options:
            convert_PCM_16 = True
        if "QUAL" in options:
            qual = True
    except:
        pass

    if to_wav:
        print("CONVERTING TO WAV")

        audio_types = ["mid", "mp3", "m4a", "ogg", "flac", "amr"]
        for file in os.listdir(test_folder_path + "/"):
            if file.split('.')[-1] in audio_types:
                sound = AudioSegment.from_file(test_folder_path + "/" + file, file.split('.')[-1])
                sound.export(test_folder_path + "/" + '.'.join(file.split('.')[0:-1]) + '.wav', format="wav")

    if norm:
        print("NORMING")

        os.system("ffmpeg-normalize " + test_folder_path + "/*.wav -f -of " + test_folder_path + "/normalized -ext wav")

    if convert_PCM_16:
        print("CONVERTING TO PCM16")

        for file in os.listdir(test_folder_path + "/normalized/"):
            if file.endswith(".wav"):
                data, samplerate = sf.read(test_folder_path + "/normalized/" + file)
                sf.write(test_folder_path + "/normalized/" + file, data, samplerate, subtype='PCM_16')

    playlist = []
    for file in os.listdir(test_folder_path + "/normalized/"):
        if file.endswith(".wav"):
            y, sr = librosa.load(test_folder_path + "/normalized/" + file)
            playlist.append((file, (y, sr)))

    permutations = {}
    for a in range(0, len(playlist) - 1):
        i = playlist[a]
        for b in range(a + 1, len(playlist)):
            j = playlist[b]

            print('Currently PyMashing: ' + i[0] + ' and ' + j[0])

            if qual:
                results = find_best_overlay(test_folder_path + "/normalized/" + i[0], i[1], test_folder_path + "/normalized/" + j[0], j[1], i[0], j[0])
                for result in results:
                    permutations[result[0]] = result[1]
            else:
                find_best_overlay(test_folder_path + "/normalized/" + i[0], i[1], test_folder_path + "/normalized/" + j[0], j[1], i[0], j[0])
      
    # write out qualities of mashups generated
    if qual:          
        with open(test_folder_path + '/qualities.txt', 'w') as f:
            f.write('')
        with open(test_folder_path + '/qualities.txt', 'a') as f:
            max_mashup = ""
            max_quality = -1
            for file in permutations:
                f.write(file + ': ' + str(permutations[file]) + '\n')
                if permutations[file] > max_quality:
                    max_mashup = file;
                    max_quality = permutations[file]

        print("Best mashup:", max_mashup, "(" + max_quality + ")")

