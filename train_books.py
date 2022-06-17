import pickle

import pandas as pd
import scipy.sparse
import scipy.sparse

from lightfm import LightFM

df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-books/5-core/train.parquet')
test_df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-books/5-core/test.parquet')

interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/books_interactions.npz')
test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/books_test_interactions.npz')
dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/books_dataset.pkl')

model = LightFM(no_components=100, loss='warp', learning_schedule='adadelta')
num_epochs = [5, 25, 50, 250, 500, 1000]
remaining = [num_epochs[0]] + [num_epochs[i] - num_epochs[i-1] for i in range(1, len(num_epochs))]
for i, epochs in enumerate(num_epochs):
    model.fit_partial(interactions, verbose=True, epochs=remaining[i], num_threads=8)
    pickle.dump(model, open(f'/pio/scratch/1/i313924/data/lightfm_data/model_{epochs}_epochs_books_no_regularize.pkl', 'wb'), protocol=4)