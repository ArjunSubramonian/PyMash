# PyMash
Creates mashups of songs and evaluates which two songs make the best mashup.

# Instructions
To mashup songs:
1) Place all songs in one directory
2) Run `python run_pymash.py (name_of_directory) (WAV) (NORM) (PCM16) (QUAL)`
- name_of_directory (where songs are located) is required
- use WAV if songs are not .wav files
- use NORM if songs are not normalized to have same perceived volume
- use PCM16 to encode songs in PCM16 (usually required)
- use QUAL to obtain quality assessment of each mashup (quality assessment takes a little while to run)
3) Mashups should now be in directory
4) If QUAL was used, look for mashup with largest number in *quality.txt* -- this should be the best mashup (or at least the best in a scentific, pop-music sense)
