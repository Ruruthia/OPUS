import pickle

import numpy as np
import pandas as pd
import scipy
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from lightfm import LightFM
from scipy.sparse import coo_matrix


def calculate_hit_rate(model, item_mapping, user_mapping, test_interactions, train_interactions):

    def hit_rate(results_pdf, k=50):
        return np.average(
            results_pdf.groupby('reviewerID').apply(
                lambda pdf: np.any(pdf['rank'] <= k)
            )
        )

    predictions = model.predict_rank(test_interactions, train_interactions, num_threads=8)
    predictions = coo_matrix(predictions)
    predictions_pdf = pd.DataFrame(
        np.vstack((predictions.row, predictions.col, predictions.data)).T,
        columns=['reviewerID', 'asin', 'rank'],
        dtype=int
    )
    predictions_pdf['reviewerID'] = predictions_pdf['reviewerID'].map(user_mapping)
    predictions_pdf['asin'] = predictions_pdf['asin'].map(item_mapping)

    return hit_rate(predictions_pdf)



def hyperopt_train_test(params):
    model = LightFM(loss='warp', **params)
    best_hit_rate = 0.0


    for _ in range(5):
        model.fit_partial(train_interactions, epochs=200, num_threads=8, word2vec_item_embeddings=None)
        current_hit_rate = calculate_hit_rate(model, item_mapping, user_mapping, test_interactions, train_interactions)
        if current_hit_rate > best_hit_rate:
            best_hit_rate = current_hit_rate
    return  best_hit_rate

def f(params):
    hit_rate = hyperopt_train_test(params)
    return {'loss': -hit_rate, 'status': STATUS_OK}

if __name__ == '__main__':
    train_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_interactions.npz')
    test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_test_interactions.npz')
    dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/5_core_dataset.pkl')
    word2vec_embeddings = pd.read_parquet('/pio/scratch/1/i313924/data/word2vec/word2vec_240_embeddings.parquet')

    word2vec_embeddings = word2vec_embeddings.rename(index=dataset.mapping()[2])
    word2vec_embeddings = word2vec_embeddings.sort_index()

    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    user_mapping = {v: k for k, v in dataset.mapping()[0].items()}

    space = {
        # for word2vec we need 240
        "no_components": hp.choice("no_components", [250]),
        "learning_schedule": hp.choice("learning_schedule", ["adagrad"]),
        "learning_rate": hp.uniform("learning_rate", 0.005, 0.1),
        # item_alpha leads to model divergence sometimes, better not to use it
        "item_alpha": hp.uniform("item_alpha", 0.0, 1e-4),
        "user_alpha": hp.uniform("user_alpha", 0.0, 1e-4),
        #"max_sampled": scope.int(hp.quniform("max_sampled", 5, 15, 1)),
    }

    trials = Trials()
    fmin(f, space, algo=tpe.suggest, max_evals=50, trials=trials)
    with open('/pio/scratch/1/i313924/data/lightfm_data/hyperopt_trials_adagrad_final.pickle', 'wb') as f_out:
        pickle.dump(trials, f_out)