# Script for training LightFM model based on Amazon Clothes

import pickle

import pandas as pd
import scipy.sparse
import scipy.sparse
from sklearn import preprocessing

from lightfm import LightFM


if __name__ == '__main__':

    df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/train.parquet')
    test_df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/test.parquet')
    categories = pd.read_json('/pio/scratch/1/recommender_systems/interim/Amazon/meta_Clothing_Shoes_and_Jewelry_categories.json', lines=True)

    interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_interactions.npz')
    test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_test_interactions.npz')
    dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/5_core_dataset.pkl')

    # For pushing clusters
    # items = df['asin'].drop_duplicates()
    # item_mapping = dataset.mapping()[2]
    # items_categories = categories[categories.asin.isin(items)].drop_duplicates()
    # without_category = set(items.values) - set(items_categories.asin)
    #
    # le = preprocessing.LabelEncoder()
    # unique_categories = items_categories.category_1.unique()
    # le.fit(unique_categories)
    #
    # items_categories.category_1 = le.transform(items_categories.category_1)
    # for item in without_category:
    #     items_categories.loc[item_mapping[item]] = pd.Series({'category_1': -1, 'asin': item})
    # items_categories.index = items_categories.asin.map(lambda x: item_mapping[x])
    # items_categories = items_categories.sort_index()
    # item_category_mapping = items_categories.category_1
    #
    model = LightFM(
        no_components=240,
        loss='warp',
        learning_schedule='adadelta',
        epsilon=2.45e-07,
        rho=0.958,
        # item_alpha=5.97e-05,
        user_alpha=2.06e-6,
        max_sampled=9,
        # num_categories=len(unique_categories),
        # item_category_mapping=item_category_mapping
    )

    num_epochs = [5, 25, 50, 250, 500, 1000]
    remaining = [num_epochs[0]] + [num_epochs[i] - num_epochs[i-1] for i in range(1, len(num_epochs))]
    for i, epochs in enumerate(num_epochs):
        model.fit_partial(interactions, verbose=True, epochs=remaining[i], num_threads=8)
        pickle.dump(model, open(f'/pio/scratch/1/i313924/data/lightfm_data/model_{epochs}_epochs_optimized_alpha_0.pkl', 'wb'), protocol=4)