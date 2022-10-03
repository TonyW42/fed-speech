from sentence_transformers import SentenceTransformer, util
import pandas as pd 

speeches = pd.read_csv("data/speech_with_description.csv")
speech   = speeches["text"].values

embedding_model = SentenceTransformer("all-mpnet-base-v2")

def sBERT_encode_all(speeches):
  encoded = []
  count = 0
  tag = []
  collapsed_tag = []
  num_main_pred = []
  group = []
  for i in range(0, speeches.shape[0]):
    text = speeches["text"].values[i]
    text_encoded = embedding_model.encode(text)

    ## append encoded
    encoded.append(text_encoded)

    ## append tag
    tag.append(samp["tag"])
    tag.append(samp["tag"])

    ## append collapse tag
    collapsed_tag.append(samp["collapsed_tag"])
    collapsed_tag.append(samp["collapsed_tag"])

    ## append num_main_ored
    num_main_pred.append(samp["num_main_preds"])
    num_main_pred.append(samp["num_main_preds"])

    ## append group category 
    group.append(0)
    group.append(1)



    if count % 50 == 0: print(f"Finished {count} / {len(data)}")
    count += 1
  return encoded, tag, collapsed_tag, num_main_pred

encoded_all, tag, collapsed_tag, num_main_pred = sBERT_encode_all(speech)



