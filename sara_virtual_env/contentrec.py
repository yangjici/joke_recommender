import pandas as pd
import time
import redis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def info(msg):
    current_app.logger.info(msg)


class ContentEngine(object):

    SIMKEY = 'p:smlr:%s'

    def __init__(self):
        self._r = redis.StrictRedis.from_url(current_app.config['REDIS_URL'])

    def train(self, data_source):
        ds = pd.read_csv(data_source)
        self._r.flushdb()

        self._train(ds)

    def _train(self, ds):
        tf = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 3),
                             min_df=0,
                             stop_words='english')
        tfidf_matrix = tf.fit_transform(ds['description'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        for idx, row in ds.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], ds['id'][i])
                             for i in similar_indices]

            flattened = sum(similar_items[1:], ())
            self._r.zadd(self.SIMKEY % row['id'], *flattened)
    def predict(self, item_id, num):
        return self._r.zrange(self.SIMKEY % item_id,
                              0,
                              num-1,
                              withscores=True,
                              desc=True)

if __name__ == "__main__":
    sample_sub_fname = "data/sample_submission.csv"
    ratings_data_fname = "data/ratings.dat"
    output_fname = "data/test_ratings.csv"

    ratings = gl.SFrame(ratings_data_fname, format='tsv')
    sample_sub = pd.read_csv(sample_sub_fname)
    for_prediction = gl.SFrame(sample_sub)
    rec_engine = ContentEngine()
    sample_sub.rating = rec_engine.predict(for_prediction)
    sample_sub.to_csv(output_fname, index=False)
