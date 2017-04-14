import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphlab as gl

# conda -V
# conda create -n jokeenv python = 2.7 anaconda
# source activate jokeenv
# conda install -n jokeenv graphlab
# source deactivate

ratings = gl.SFrame(ratings_data_fname, format='tsv')
sample_sub = pd.read_csv(sample_sub_fname)
for_prediction = gl.SFrame(sample_sub)

jokes = gl.SFrame('data/jokes.dat', format='tsv')
