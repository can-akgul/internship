import numpy as np
import pandas as pd

data = pd.read_csv("liar_dataset/train.csv", sep=',', header=None, usecols=[1, 2], names=["label", "statement"])

data['label'] = data['label'].apply(lambda x:1 if x in ["false", "pants-fire", "barely-true"] else 0)

print(data)

