import pandas as pd
import graphlab as gl

if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    jokes_fname =
    ratings_data_fname = "data/ratings.dat"
    output_fname = "data/test_ratings.csv"

    #split out jokes by piece
    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    jokes = gl.SFrame(‘data/jokes.dat’, format=‘tsv’)
    all_jokes = jokes.to_numpy().tolist()
    merged=“”.join(list(itertools.chain(*all_jokes)))
    jokes_index = np.array(re.split(‘</p>’,merged)[:150])

    #create bag of words
    new_bow = []
    for n in jokes_index:
        new_bow.append(gl.text_analytics.count_words(sf['n']))
    #tfidf of bag of words
    sf['tfidf'] = gl.text_analytics.tf_idf(sf['new_bol'])

    #predict
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    rec_engine = gl.recommender.item_content_recommender.create(item_data='jokes',
                                                     user_id="user_id",
                                                     item_id="joke_id",
                                                     target='rating')

    sample_sub.rating = rec_engine.predict(for_prediction)
    sample_sub.to_csv(output_fname, index=False)
