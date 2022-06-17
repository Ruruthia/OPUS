import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

if __name__ == "__main__":
    model = pd.read_pickle('/pio/scratch/1/i313924/data/lightfm_data/warp_model_1000_epochs_slim.pkl')
    item_embeddings = pd.DataFrame(model.item_embeddings)

    tsne = TSNE(init='pca', verbose=10)

    item_embeddings_transformed = tsne.fit_transform(item_embeddings.values)

    np.save('/pio/scratch/1/i313924/data/lightfm_data/tsne_embeddings_1000_epochs_slim.npy', item_embeddings_transformed)

