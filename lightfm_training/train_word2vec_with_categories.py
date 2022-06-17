# Script for training LightFM model initialized with word2vec embeddings, which include item features embeddings

import pickle

import pandas as pd
import scipy.sparse
import scipy.sparse
from lightfm import LightFM

if __name__ == '__main__':

    df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/train.parquet')
    test_df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/test.parquet')
    categories = pd.read_json('/pio/scratch/1/recommender_systems/interim/Amazon/meta_Clothing_Shoes_and_Jewelry_categories.json', lines=True)

    interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_interactions.npz')
    test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_test_interactions.npz')
    dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/5_core_dataset_features.pkl')
    item_features = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/5_core_features.pkl')

    word2vec_embeddings = pd.read_parquet('/pio/scratch/1/i313924/data/word2vec/word2vec_240_embeddings_categories.parquet')
    item_mapping = dataset.mapping()[3]
    word2vec_embeddings = word2vec_embeddings.rename(index=item_mapping)
    word2vec_embeddings = word2vec_embeddings.sort_index()

    model = LightFM(
        no_components=240,
        loss='warp',
        learning_schedule='adadelta',
        epsilon=2.45e-07,
        rho=0.958,
        # item_alpha=5.97e-05,
        user_alpha=2.06e-06,
        max_sampled=9
    )

    num_epochs = [5, 25, 50, 250, 500, 1000]
    remaining = [num_epochs[0]] + [num_epochs[i] - num_epochs[i-1] for i in range(1, len(num_epochs))]
    for i, epochs in enumerate(num_epochs):
        model.fit_partial(interactions, item_features=item_features, verbose=True, epochs=remaining[i], num_threads=8, word2vec_embeddings=word2vec_embeddings.values)
        pickle.dump(model, open(f'/pio/scratch/1/i313924/data/lightfm_data/model_{epochs}_word2vec_categories_item_alpha_0.pkl', 'wb'), protocol=4)