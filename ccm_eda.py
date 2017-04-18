import pandas as pd
import numpy as np
import graphlab as gl
from sklearn.feature_extraction.text import TfidfVectorizer
import itertools
import re



jokes = gl.SFrame('data/jokes.dat', format='tsv')
ratings = gl.SFrame('data/ratings.dat', format='tsv')

ratings_df = gl.SFrame.to_dataframe(ratings)

all_jokes = jokes.to_numpy().tolist()
all_jokes = [item for sublist in all_jokes for item in sublist]
clean_jokes = []
marks = ['<p>', '/>','</p>']
for joke in all_jokes:
    if joke.split()[-1] in marks:
        pass
    else:
        clean_jokes.append(joke)

        jokes = gl.SFrame('data/jokes.dat', format='tsv')

all_jokes = jokes.to_numpy().tolist()



merged="".join(list(itertools.chain(*all_jokes)))

jokes_index = np.array(re.split('</p>',merged)[:150])

jokes_denum = []

for joke in jokes_index:
    no_num = "".join([w for w in joke if not w.isdigit()])
    jokes_denum.append(no_num)


jokes_denum = np.array(jokes_denum)

vector= TfidfVectorizer(max_features=100, stop_words = 'english')

joke_feature_space = vector.fit_transform(jokes_denum)


joke_feature_space = joke_feature_space.toarray()

df_joke = pd.DataFrame(joke_feature_space)

df_joke['joke_id'] = xrange(1,len(df_joke)+1)

jokes_gl = gl.SFrame(df_joke)

# jokes = gl.SFrame('data/jokes.dat', format='tsv')
# all_jokes = jokes.to_numpy().tolist()
# merged="".join(list(itertools.chain(*all_jokes)))
# jokes_index = np.array(re.split('</p>', merged)[:150])


(train_ratings, test_ratings) = ratings.random_split(0.8)

#isr = gl.recommender.item_similarity_recommender.ItemSimilarityRecommender()

# sim_rec = gl.recommender.item_similarity_recommender.create(train_ratings, user_id='user_id', item_id='joke_id', target='rating')
#
# sim_rec.evaluate(train_ratings, 'rmse', target='rating')

# sim_rec2 = gl.recommender.item_similarity_recommender.create(train_ratings, user_id='user_id', item_id='joke_id', target='rating', similarity_type='cosine')
#
# sim_rec2.evaluate_rmse(train_ratings, target='rating')
#
# # pred2 = sim_rec2.predict(test_ratings)
#
# nn = sim_rec2.get_similar_items()
#
# cossim_rec = gl.recommender.item_similarity_recommender.create(train_ratings, user_id='user_id', item_id='joke_id', target='rating', similarity_type='cosine', nearest_items=nn)
#
# cossim_rec.evaluate_rmse(train_ratings, target='rating')


sim_rec3 = gl.recommender.item_similarity_recommender.create(train_ratings, user_id='user_id', item_id='joke_id', target='rating', similarity_type='pearson', item_data=jokes_gl)

sim_rec3.evaluate_rmse(train_ratings, target='rating')

uu = sim_rec3.get_similar_users()
nn = sim_rec3.get_similar_items()

pearsim_rec = gl.recommender.item_similarity_recommender.create(train_ratings, user_id='user_id', item_id='joke_id', target='rating', similarity_type='cosine', nearest_items=nn, item_data=jokes_gl, user_data=uu)

pearsim_rec.evaluate_rmse(train_ratings, target='rating')

################################

baseline_rmse = gl.evaluation.rmse(test_ratings['rating'], sim_rec3.predict(test_ratings)
print baseline_rmse

#################################

rec_engine = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating',
                                                     solver='auto',
                                                     num_factors = 30,
                                                     ranking_regularization = 0
                                                     )
