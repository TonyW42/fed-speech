from sentence_transformers import SentenceTransformer, util
import pandas as pd 
import numpy as np 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

speeches = pd.read_csv("data/speech_with_description.csv")
# speech   = speeches["text"].values[1]

embedding_model = SentenceTransformer("all-mpnet-base-v2")

def sBERT_encode_all(speeches):
  encoded = []
  count = 0

  for i in range(0, speeches.shape[0]):
    text = speeches["text"].values[i]
    text_encoded = embedding_model.encode(text)

    ## append encoded
    encoded.append(text_encoded)


    if count % 50 == 0: print(f"Finished {count} / {speeches.shape[0]}")
    count += 1
  return encoded

encoded_all = sBERT_encode_all(speeches)
encoded_df = np.array(encoded_all)

transformed_embedding = TSNE(random_state = 42).fit_transform(encoded_df)

df = pd.DataFrame(
    dict(x_0 = transformed_embedding[:, 0], 
         x_1 = transformed_embedding[:, 1], 
         consumption = speeches["consumption"].values,
         economic_activity = speeches["economic_activity"].values,
         inflation = speeches["inflation"].values,
         inflation_binary = speeches["inflation_binary"].values,
         unemployment = speeches["unemployment"].values,
         )
)

def plot_tsne(color_by = "consumption"):
    plt.figure()
    if color_by is not None:
      my_plot = sns.scatterplot("x_0", "x_1", hue = color_by, data= df).set(title = f"Two-dimensional Representation for Speeches, colored by {color_by}")
      plt.savefig(f"figures/tsne_plot_by_{color_by}")
    else:
      my_plot = sns.scatterplot("x_0", "x_1", data= df).set(title = f"Two-dimensional Representation for Speeches")
      plt.savefig(f"figures/tsne_plot")

if __name__ == "__main__":
    plot_tsne(color_by = None)
    plot_tsne(color_by = "consumption")
    plot_tsne(color_by = "economic_activity")
    plot_tsne(color_by = "inflation")
    plot_tsne(color_by = "inflation_binary")
    plot_tsne(color_by = "unemployment")



