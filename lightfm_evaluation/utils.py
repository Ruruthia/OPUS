import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import spatial
from scipy.sparse import coo_matrix
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

def hit_rate(results_pdf, k=50):
    return np.average(
        results_pdf.groupby('reviewerID').progress_apply(
            lambda pdf: np.any(pdf['rank'] <= k)
        )
    )

def precision(results_pdf, k=50):
    return np.average(
        results_pdf.groupby('reviewerID').progress_apply(
            lambda pdf: np.sum(pdf['rank'] <= k) / k
        )
    )

def recall(results_pdf, k=50):
    return np.average(
        results_pdf.groupby('reviewerID').progress_apply(
            lambda pdf: np.sum(pdf['rank'] <= k) / len(pdf)
        )
    )

def calculate_metrics(results, user_mapping, item_mapping):
    results = coo_matrix(results)
    results_pdf = pd.DataFrame(
        np.vstack((results.row, results.col, results.data)).T,
        columns=['reviewerID', 'asin', 'rank'],
        dtype=int
    )
    results_pdf['reviewerID'] = results_pdf['reviewerID'].map(user_mapping)
    results_pdf['asin'] = results_pdf['asin'].map(item_mapping)

    return hit_rate(results_pdf), recall(results_pdf), precision(results_pdf)

def analyse_metrics(models, train_interactions, test_interactions, dataset, item_features=None):

    test_precision_scores = []
    test_recall_scores = []
    test_hit_rate_scores = []

    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    user_mapping = {v: k for k, v in dataset.mapping()[0].items()}
    for model in tqdm(models):

        results = model.predict_rank(test_interactions, train_interactions, num_threads=8, item_features=item_features)
        hr, r, p = calculate_metrics(results, user_mapping, item_mapping)

        test_hit_rate_scores.append(hr)
        test_recall_scores.append(r)
        test_precision_scores.append(p)

    return test_hit_rate_scores, test_recall_scores, test_precision_scores

def analyse_ABX(path, items_embeddings, pca=None):
    abx_tests = pd.read_json(path, lines=True)
    lines = len(abx_tests)

    A = items_embeddings.loc[abx_tests["A"]].values
    B = items_embeddings.loc[abx_tests["B"]].values
    X = items_embeddings.loc[abx_tests["X"]].values

    dist_A = ((A - X)**2).sum(axis=1)
    dist_B = ((B - X)**2).sum(axis=1)

    cos_dist_A = np.zeros(lines)
    cos_dist_B = np.zeros(lines)

    for i in range(lines):
        cos_dist_A[i] = spatial.distance.cosine(A[i, :], X[i, :])
        cos_dist_B[i] = spatial.distance.cosine(B[i, :], X[i, :])

    if pca is not None:
        pca_A = pca.transform(A)
        pca_B = pca.transform(B)
        pca_X = pca.transform(X)

        dist_pca_A = ((pca_A - pca_X)**2).sum(axis=1)
        dist_pca_B  = ((pca_B - pca_X)**2).sum(axis=1)

        cos_dist_pca_A = np.zeros(lines)
        cos_dist_pca_B = np.zeros(lines)

        for i in range(lines):
            cos_dist_pca_A[i] = spatial.distance.cosine(pca_A[i, :], pca_X[i, :])
            cos_dist_pca_B[i] = spatial.distance.cosine(pca_B[i, :], pca_X[i, :])

    return [(dist_A < dist_B).mean(),
            0 if pca is None else (dist_pca_A < dist_pca_B).mean(),
            0 if pca is None else ((dist_A < dist_B) == (dist_pca_A < dist_pca_B)).mean()],\
           [(cos_dist_A < cos_dist_B).mean(),
            0 if pca is None else (cos_dist_pca_A < cos_dist_pca_B).mean(),
            0 if pca is None else ((cos_dist_A < cos_dist_B) == (cos_dist_pca_A < cos_dist_pca_B)).mean()]

def reduce_item_embeddings(model, dataset, categories, pca=None, item_features=None):
    item_embeddings = pd.DataFrame(model.get_item_representations(item_features)[1])
    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(item_embeddings)
    reduced_item_embeddings = pd.DataFrame(pca.transform(item_embeddings))
    reduced_item_embeddings = reduced_item_embeddings.rename(index=item_mapping)
    reduced_item_embeddings = reduced_item_embeddings.join(categories.set_index('asin'))
    reduced_item_embeddings.columns = ['0', '1', 'category_1', 'category_2']
    return reduced_item_embeddings

def analyse_embeddings(num_epochs, models, dataset, categories, abx_path, women_ids=None, men_ids=None, item_features=None):

    item_embeddings = pd.DataFrame()
    user_embeddings = pd.DataFrame()

    for model in models:

        item_embeddings = pd.concat([item_embeddings, pd.DataFrame(model.get_item_representations(item_features)[1])])
        user_embeddings = pd.concat([user_embeddings, pd.DataFrame(model.user_embeddings)])

    # fitting PCA on all embeddings
    item_pca = PCA(n_components=2)
    item_pca.fit(item_embeddings)
    user_pca = PCA(n_components=2)
    user_pca.fit(user_embeddings)

    print("PCA prepared")

    eucl_scores = []
    cos_scores = []
    women_means = []
    men_means = []

    item_mapping = {v: k for k, v in dataset.mapping()[2].items()}

    for i, (epochs, model) in enumerate(zip(num_epochs, models)):

        figure, axis = plt.subplots(1, 2, figsize=(12,6))
        print(f"Working on model from epoch {epochs}")


        user_embeddings = pd.DataFrame(model.get_item_representations(item_features)[1])
        reduced_user_embeddings = pd.DataFrame(user_pca.transform(user_embeddings))
        axis[0].scatter(reduced_user_embeddings[0], reduced_user_embeddings[1], s=0.03)
        # axis[0].set_xlim([-1, 1])
        # axis[0].set_ylim([-1, 1])

        item_embeddings =  pd.DataFrame(model.item_embeddings)
        reduced_item_embeddings = reduce_item_embeddings(model, dataset, categories, pca=item_pca)
        sns.scatterplot(x='0', y='1', data=reduced_item_embeddings, hue='category_1', s=10, ax=axis[1], legend=False)
        # axis[1].set_xlim([-15, 15])
        # axis[1].set_ylim([-7, 7])
        figure.show()

        item_embeddings = item_embeddings.rename(index=item_mapping)

        if women_ids is not None:
            women_mean = item_embeddings.loc[women_ids].mean(axis = 0)
            women_means.append(women_mean)
        if men_ids is not None:
            men_mean = item_embeddings.loc[men_ids].mean(axis = 0)
            men_means.append(men_mean)

        # calculating ABX scores
        eucl_score, cos_score = analyse_ABX(abx_path, item_embeddings, pca = item_pca)

        eucl_scores.append(eucl_score)
        cos_scores.append(cos_score)

    return eucl_scores, cos_scores, women_means, men_means, item_pca