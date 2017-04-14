import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphlab as gl

#set up conda env
# conda -V
# conda create -n jokeenv python = 2.7 anaconda
# source activate jokeenv
# conda install -n jokeenv graphlab
# source deactivate

#import data
jokes = gl.SFrame('data/jokes.dat', format='tsv', header=None)
all_jokes = jokes.to_numpy().tolist()
ratings = gl.SFrame('data/ratings.dat', format='tsv')

#clean joke code
all_jokes = jokes.to_numpy().tolist()
all_jokes = [item for sublist in all_jokes for item in sublist]
clean_jokes = []
marks = ['<p>', '/>','</p>']
for joke in all_jokes:
    if joke.split()[-1] in marks:
        pass
    else:
        clean_jokes.append(joke)
