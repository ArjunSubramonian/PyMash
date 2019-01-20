#%% [markdown]

from __future__ import print_function, division

import thinkdsp
import thinkplot
import thinkstats2

import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')
#%%

#%% [markdown]
# The Voss algorithm is described in this paper:
# 
# Voss, R. F., & Clarke, J. (1978). "1/f noise" in music: Music from 1/f noise". *Journal of the Acoustical Society of America* 63: 258–263. Bibcode:1978ASAJ...63..258V. doi:10.1121/1.381721.
# 
# And presented by Martin Gardner in this *Scientific American* article:
# 
# Gardner, M. (1978). "Mathematical Games—White and brown music, fractal curves and one-over-f fluctuations". *Scientific American* 238: 16–32.
# 
# McCartney suggested an improvement here:
# 
# http://www.firstpr.com.au/dsp/pink-noise/
# 
# And Trammell proposed a stochastic version here:
# 
# http://home.earthlink.net/~ltrammell/tech/pinkalg.htm
#%% [markdown]
# The fundamental idea of this algorithm is to add up several sequences of random numbers that get updated at different rates.  The first source should get updated at every time step; the second source every other time step, the third source ever fourth step, and so on.
# 
# In the original algorithm, the updates are evenly spaced.  In the stochastic version, they are randomly spaced.
# 
# My implementation starts with an array with one row per timestep and one column for each of the white noise sources.  Initially, the first row and the first column are random and the rest of the array is Nan.

#%%
nrows = 100
ncols = 5

array = np.empty((nrows, ncols))
array.fill(np.nan)
array[0, :] = np.random.random(ncols)
array[:, 0] = np.random.random(nrows)
array[0:6]

#%% [markdown]
# The next step is to choose the locations where the random sources change.  If the number of rows is $n$, the number of changes in the first column is $n$, the number in the second column is $n/2$ on average, the number in the third column is $n/4$ on average, etc.
# 
# So the total number of changes in the matrix is $2n$ on average; since $n$ of those are in the first column, the other $n$ are in the rest of the matrix.
# 
# To place the remaining $n$ changes, we generate random columns from a geometric distribution with $p=0.5$.  If we generate a value out of bounds, we set it to 0 (so the first column gets the extras).

#%%
p = 0.5
n = nrows
cols = np.random.geometric(p, n)
cols[cols >= ncols] = 0
cols

#%% [markdown]
# Within each column, we choose a random row from a uniform distribution.  Ideally we would choose without replacement, but it is faster and easier to choose with replacement, and I doubt it matters.

#%%
rows = np.random.randint(nrows, size=n)
rows

#%% [markdown]
# Now we can put random values at rach of the change points.

#%%
array[rows, cols] = np.random.random(n)
array[0:6]

#%% [markdown]
# Next we want to do a zero-order hold to fill in the NaNs.  NumPy doesn't do that, but Pandas does.  So I'll create a DataFrame:

#%%
df = pd.DataFrame(array)
df.head()

#%% [markdown]
# And then use `fillna` along the columns.

#%%
filled = df.fillna(method='ffill', axis=0)
filled.head()

#%% [markdown]
# Finally we add up the rows.

#%%
total = filled.sum(axis=1)
total.head()

#%% [markdown]
# If we put the results into a Wave, here's what it looks like:

#%%
wave = thinkdsp.Wave(total.values)
wave.plot()

#%% [markdown]
# Here's the whole process in a function:

#%%
def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.
    
    nrows: number of values to generate
    rcols: number of random sources to add
    
    returns: NumPy array
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)
    
    # the total number of changes is nrows
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)

    return total.values

#%% [markdown]
# To test it I'll generate 11025 values:

#%%
ys = voss(11025)
ys

#%% [markdown]
# And make them into a Wave:

#%%
wave = thinkdsp.Wave(ys)
wave.unbias()
wave.normalize()

#%% [markdown]
# Here's what it looks like:

#%%
wave.plot()
thinkplot.config(xlabel='Time')

#%% [markdown]
# As expected, it is more random-walk-like than white noise, but more random looking than red noise.
# 
# Here's what it sounds like:

#%%
wave.make_audio()

#%% [markdown]
# And here's the power spectrum:

#%%
spectrum = wave.make_spectrum()
spectrum.hs[0] = 0
spectrum.plot_power()
thinkplot.config(xlabel='Frequency (Hz)',
                 xscale='log', 
                 yscale='log')

#%% [markdown]
# The estimated slope is close to -1.

#%%
spectrum.estimate_slope().slope

#%% [markdown]
# We can get a better sense of the average power spectrum by generating a longer sample:

#%%
seg_length = 64 * 1024
iters = 100
wave = thinkdsp.Wave(voss(seg_length * iters))
len(wave)

#%% [markdown]
# And using Barlett's method to compute the average.

#%%
def bartlett_method(wave, seg_length=512, win_flag=True):
    """Estimates the power spectrum of a noise wave.
    
    wave: Wave
    seg_length: segment length
    """
    # make a spectrogram and extract the spectrums
    spectro = wave.make_spectrogram(seg_length, win_flag)
    spectrums = spectro.spec_map.values()
    
    # extract the power array from each spectrum
    psds = [spectrum.power for spectrum in spectrums]
    
    # compute the root mean power (which is like an amplitude)
    hs = np.sqrt(sum(psds) / len(psds))
    fs = spectrums[0].fs
    
    # make a Spectrum with the mean amplitudes
    spectrum = thinkdsp.Spectrum(hs, fs, wave.framerate)
    return spectrum


#%%
spectrum = bartlett_method(wave, seg_length=seg_length, win_flag=False)
spectrum.hs[0] = 0
len(spectrum)

#%% [markdown]
# It's pretty close to a straight line, with some curvature at the highest frequencies.

#%%
spectrum.plot_power()
thinkplot.config(xlabel='Frequency (Hz)',
                 xscale='log', 
                 yscale='log')

#%% [markdown]
# And the slope is close to -1.

#%%
spectrum.estimate_slope().slope


#%%



