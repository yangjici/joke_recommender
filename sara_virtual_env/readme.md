# Recommend a good joke.

Today you are going to have a little friendly competition with your classmates. 
You are going to building a recommendation system based off data from the 
[Jester Dataset](http://eigentaste.berkeley.edu/dataset/). It includes user 
ratings of over 100 jokes.

The joke texts are in `data/jokes.dat`. Please forgive how awful they all are.
We did not write them, and they do not represent the views of the Galvanize staff. The users'
ratings have been broken into a training and test set for you, with the training 
data in `data/ratings.dat`. Your goal is to build a recommendation system 
and to suggest jokes to users! Your score will be measured based off of how well 
you predict the top-rated jokes for the users' ratings in our test set. 

Note that we will be using [GraphLab](https://turi.com/) to build our 
recommendation system. If you don't already have it installed, follow the 
directions in [GraphLab_setup.md](GraphLab_setup.md).

To make sure that you have it set up correctly, run `src/rec_runner.py`. If it 
completes without error, you are good to go! Note that `src/rec_runner.py` 
outputs a _properly formatted_ file of recommendations for you! I suggest 
using this file as a base and building off of it as you work through the day. 

Next, make sure that you are able to post your score to Slack. To begin, you'll
need to pip install the `performortron` library: 

```bash 
pip install git+https://github.com/zipfian/performotron.git --upgrade
```

After that, you should be able to use the `src/slack_poster.py` file to 
post your score to slack. The `src/slack_poster.py` file takes a _properly 
formatted_ file of recommendations (see `data/sample_submission.csv` for an 
example) and reports your score to Slack. Use the URL in `src/config.yaml` for 
the URL that `slack_poster` will prompt you for, and use a gschool channel to 
post results to, **prefacing the channel with a `#` when you are promted 
(i.e. #dsi)**. Test it out with the following command:
    
```bash
python src/slack_poster.py data/sample_submission.csv
```

See if you can get the best score! 

# Evaluation

## Score

For each user, our scoring metric will select the 5% of jokes you thought would be most highly rated by that user. It then looks at the actual ratings (in the test data) that the user gave those jokes.  Your score is the average of those ratings.

Thus, for an algorithm to score well, it only needs to identify which jokes a user is likely to rate most highly (so the absolute accuracy of your ratings is less important than the rank ordering).

As mentioned above, your submission should be in the same format as the sample 
submission file, and the only thing that will be changed is the ratings column. 
Use `src/rec_runner.py` as a starting point, as it has a function to create 
correctly formatted submission files.

## References

### GraphLab

* [Getting Started with GraphLab Create](http://turi.com/learn/notebooks/getting_started_with_graphlab_create.html)
* [Five Line Recommender Explained](http://turi.com/learn/notebooks/five_line_recommender.html)
* [Building a Recommender with Ratings Data](http://turi.com/learn/notebooks/recsys_explicit_rating.html)
* [Basic Recommender Functionalities](http://turi.com/learn/notebooks/basic_recommender_functionalities.html)
* [Building a Recommender with Implicit data](http://turi.com/learn/notebooks/recsys_rank_10K_song.html)


