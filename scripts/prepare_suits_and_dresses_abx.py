import random
import re

import pandas as pd
from tqdm.auto import tqdm


def find_suits_and_dresses(categories, ids_to_choose_from):
    suits = categories[categories['title'].str.contains('suit|tuxedo', regex=True, flags=re.IGNORECASE).fillna(False)]
    suits = suits[suits.category.apply(lambda x: 'Men' in x)]

    dresses = categories[categories['title'].str.contains('dress', regex=True, flags=re.IGNORECASE).fillna(False)]
    dresses = dresses[dresses.category.apply(lambda x: 'Women' in x)]

    suits = suits.loc[suits['asin'].isin(ids_to_choose_from)]
    dresses = dresses.loc[dresses['asin'].isin(ids_to_choose_from)]

    return suits, dresses


def prepare_test_file(suits, dresses, output_path):
    test_set = []
    for _ in tqdm(range(10000)):
        positive = 'Women'
        negative = 'Men'
        if random.random() < 0.5:
            positive, negative = negative, positive
        if positive == 'Women':
            positive_items = dresses.sample(2)['asin'].values
            negative_item = suits.sample(1)['asin'].values
        else:
            positive_items = suits.sample(2)['asin'].values
            negative_item = dresses.sample(1)['asin'].values

        line = {"A": positive_items[0],
                "B": negative_item[0],
                "X": positive_items[1],
                "category_AX": positive,
                "category_B": negative
                }
        test_set.append(line)
    with open(output_path, 'w') as file:
        pd.DataFrame(test_set).to_json(file, 'records', lines=True)

if __name__ == "__main__":
    # read data
    categories = pd.read_json('/pio/scratch/1/recommender_systems/interim/Amazon/meta_Clothing_Shoes_and_Jewelry_categories.json', lines=True)
    ratings = pd.read_csv('/pio/scratch/1/i313924/data/train_data/slim_ratings.csv', names=['asin', 'reviewerID', 'overall', 'unixReviewTime'])
    ids_to_choose_from = ratings['asin'].unique()

    # prepare test file
    suits, dresses = find_suits_and_dresses(categories, ids_to_choose_from)
    prepare_test_file(suits, dresses, '/pio/scratch/1/i313924/data/test_data/Suits_Dresses_ABX')

