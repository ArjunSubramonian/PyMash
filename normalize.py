import thinkdsp
import thinkplot
import thinkstats2

import numpy as np
import pandas as pd
import os
import soundfile as sf

def normalize(path_to_wav_file):
    f = thinkdsp.read_wave(path_to_wav_file)
    f.normalize()
    f.write(filename=path_to_wav_file)

if __name__ == "__main__":
    for file in os.listdir("permutations/"):
        if (file.endswith(".wav")):
            data, samplerate = sf.read("permutations/" + file)
            sf.write("permutations/" + file, data, samplerate, subtype='PCM_16')
            normalize("permutations/" + file)
