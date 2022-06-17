import pickle

import pandas as pd
import scipy.sparse
from lightfm import LightFM
from lightfm.data import Dataset

THREADS = 8

df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/train.parquet')
test_df = pd.read_parquet('/pio/scratch/1/recommender_systems/processed/amazon-clothes/5-core/test.parquet')
categories = pd.read_json('/pio/scratch/1/recommender_systems/interim/Amazon/meta_Clothing_Shoes_and_Jewelry_categories.json', lines=True)

interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_interactions.npz')
test_interactions = scipy.sparse.load_npz('/pio/scratch/1/i313924/data/lightfm_data/5_core_test_interactions.npz')
dataset = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/5_core_dataset.pkl')

items_list = df.asin.unique()
categories  = categories[categories.asin.isin(items_list)]
categories_names = categories.category_1.unique()

dataset.fit_partial(item_features=(x for x in categories_names))
with open('/pio/scratch/1/i313924/data/lightfm_data/5_core_dataset_features.pkl', 'wb') as f:
    pickle.dump(dataset, f, -1)

item_features = dataset.build_item_features((row.asin, [row.asin, row.category_1])
                                            for _, row in categories.iterrows())
with open('/pio/scratch/1/i313924/data/lightfm_data/5_core_features.pkl', 'wb') as f:
    pickle.dump(item_features, f, -1)

model = LightFM(
    no_components=240,
    loss='warp',
    learning_schedule='adadelta',
    epsilon=2.45e-07,
    rho=0.958,
    item_alpha = 5.97e-05,
    user_alpha=2.06e-06,
    max_sampled=9
)

num_epochs = [5, 25, 50, 250, 500, 1000]
remaining = [num_epochs[0]] + [num_epochs[i] - num_epochs[i-1] for i in range(1, len(num_epochs))]
for i, epochs in enumerate(num_epochs):
    model.fit_partial(interactions, item_features=item_features, verbose=True, epochs=remaining[i], num_threads=8)
    pickle.dump(model, open(f'/pio/scratch/1/i313924/data/lightfm_data/model_{epochs}_epochs_optimal_hyperparams_categories_item_alpha_5.97e-05.pkl', 'wb'), protocol=4)