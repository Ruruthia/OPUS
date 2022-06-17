import random

import pandas as pd
from tqdm.auto import tqdm

DATA_PATH = '/pio/scratch/1/recommender_systems'

def prepare_test_file(categories, output_path, test_set_size=10000, category_type="category_1",
                      weighted_sample=True, min_counts=0, items_to_chose_from=None):

    counts = categories[category_type].value_counts()
    counts = counts[counts > min_counts]
    categories_grouped = categories.groupby(category_type)

    categories_dict = {}
    for category, items in categories_grouped:
        if items_to_chose_from is not None:
            categories_dict[category] = items.loc[items['asin'].isin(items_to_chose_from)]
        else:
            categories_dict[category] = items

    test_set = []
    for _ in tqdm(range(test_set_size)):

        # choosing categories for ABX
        if weighted_sample:
            category_sample = counts.sample(2, weights=counts)
        else:
            category_sample = counts.sample(2)

        # choosing which category to treat as positive & negative
        positive = category_sample.index[0]
        negative = category_sample.index[1]
        if random.random() < 0.5:
            positive, negative = negative, positive

        # choosing items for ABX from positive & nega
        positive_items = categories_dict[positive].sample(2).asin.values
        negative_item = categories_dict[negative].sample(1).asin.values

        # appending record
        line = {"A": positive_items[0],
                "B": negative_item[0],
                "X": positive_items[1],
                "category_AX": positive,
                "category_B": negative}
        test_set.append(line)

    with open(output_path, 'w') as file:
        pd.DataFrame(test_set).to_json(file, 'records', lines=True)


if __name__ == "__main__":
    # read & prepare data
    df_positive = pd.read_parquet(f'{DATA_PATH}/processed/amazon-books/5-core/train.parquet')
    df_negative = pd.read_parquet(f'{DATA_PATH}/processed/amazon-books/5-core/negative_train.parquet')

    categories = pd.read_parquet(f'{DATA_PATH}/interim/Amazon/meta_Books_categories.parquet')
    categories.category_1 = categories.category_1.map(lambda s: s.replace('&amp;', '&'))

    # prepare ids to chose from
    df = pd.concat([df_positive, df_negative])
    items_to_chose_from = df.asin.unique()

    # prepare test file
    prepare_test_file(categories, f'{DATA_PATH}/interim/ABX_tests/books.json',
                      category_type='category_1', test_set_size=10000, items_to_chose_from=items_to_chose_from)