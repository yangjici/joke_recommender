import pandas as pd
import graphlab as gl

# params = {'user_id':"user_id",
# 'item_id':"joke_id",
# 'target':'rating',
# 'solver':['auto'],
# 'ranking_regularization' : [0.2, 0.25, 0.3],
# 'num_factors' : [32,35,40,45],
# 'nmf' : True
# }
# ratings = gl.SFrame('data/ratings.dat', format='tsv')
# (train, valid) = ratings.random_split(.8)
# job = gl.toolkits.model_parameter_search.grid_search.create((train,valid), gl.ranking_factorization_recommender.create,params)

if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    output_fname = "data/test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    rec_engine = gl.ranking_factorization_recommender.create(observation_data=ratings,
                                                     user_id="user_id",num_factors=30,
                                                     item_id="joke_id",
                                                     target='rating',ranking_regularization=0,
                                                     solver='auto')

    sample_sub.rating = rec_engine.predict(for_prediction)
    sample_sub.to_csv(output_fname, index=False)
#
# #2.95 Score --> number of factors = 40, no other difference
