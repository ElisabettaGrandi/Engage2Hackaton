import pandas as pd

train = pd.read_csv("data/engage-ktn-2026-competition/train.csv")

print("Dimensioni:", train.shape)
print("Snapshot:", train["sample_id"].nunique())
print("Valori mancanti per colonna:")
print(train.isna().sum()[lambda x: x > 0])