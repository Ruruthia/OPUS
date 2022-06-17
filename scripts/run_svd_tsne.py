import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

if __name__ == "__main__":
    model = pd.read_pickle('/pio/scratch/1/i313924/data/svd_data/model_1000_epochs_regularized.pkl')
    item_embeddings = pd.DataFrame(model.qi_)

    tsne = TSNE(init='pca', verbose=10)

    item_embeddings_transformed = tsne.fit_transform(item_embeddings.values)
    np.save('/pio/scratch/1/i313924/data/svd_data/tsne_embeddings_1000_epochs_regularized.npy', item_embeddings_transformed)