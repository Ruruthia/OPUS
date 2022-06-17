# Script for training LightFM model based on Amazon Books


import pickle

import pandas as pd
import scipy.sparse
import scipy.sparse

from lightfm import LightFM

if __name__ == '__main__':

    df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-books/5-core/train.parquet')
    test_df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-books/5-core/test.parquet')

    interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/books_interactions.npz')
    test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/books_test_interactions.npz')
    dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/books_dataset.pkl')

    # For pushing clusters
    # categories = pd.read_parquet('/pio/scratch/1/recommender_systems/interim/Amazon/meta_Books_categories.parquet')
    # categories.category_1 = categories.category_1.map(lambda s: s.replace('&amp;', '&'))
    # items = df['asin'].drop_duplicates()
    # item_mapping = dataset.mapping()[2]
    # items_categories = categories[categories.asin.isin(items)].drop_duplicates()
    # without_category = set(items.values) - set(items_categories.asin)
    #
    # le = preprocessing.LabelEncoder()
    # unique_categories = items_categories.category_1.unique()
    # le.fit(unique_categories)
    #
    # to_add = pd.Series([item for item in list(without_category)], name='asin')
    # to_add = pd.DataFrame(to_add)
    # to_add['category_1'] = -1
    #
    # items_categories.category_1 = le.transform(items_categories.category_1)
    # items_categories = pd.concat([items_categories, to_add])
    # items_categories.index = items_categories.asin.map(lambda x: item_mapping[x])
    # items_categories = items_categories.sort_index()
    # item_category_mapping = items_categories.category_1

    model = LightFM(no_components=100, loss='warp', learning_schedule='adadelta')
    # model = LightFM(no_components=100, loss='warp', learning_schedule='adadelta', num_categories=len(unique_categories), item_category_mapping=item_category_mapping)
    num_epochs = [5, 25, 50, 250, 500, 1000]
    remaining = [num_epochs[0]] + [num_epochs[i] - num_epochs[i-1] for i in range(1, len(num_epochs))]
    for i, epochs in enumerate(num_epochs):
        model.fit_partial(interactions, verbose=True, epochs=remaining[i], num_threads=8)
        pickle.dump(model, open(f'/pio/scratch/1/i313924/data/lightfm_data/model_{epochs}_epochs_books_no_regularize.pkl', 'wb'), protocol=4)