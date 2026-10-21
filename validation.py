import pandas as pd
from xgboost import XGBRanker, XGBClassifier
from preprocessing import preprocess_memory_efficient

# 1. Caricamento del dataset di validation
df = pd.read_csv("data/engage-ktn-2026-competition/validation.csv")

# 2. Preprocessing (lo stesso utilizzato nel training)
X_val, y_val, qid_val = preprocess_memory_efficient(df)

# 3. Recuperiamo i veri row_index, rispettando l'ordinamento del preprocessing
row_indices = df.sort_values("sample_id")["row_index"].to_numpy()

# 4. Caricamento del modello già addestrato
model = XGBClassifier()
#model = XGBRanker()
model.load_model("xgboost_classifier.json")#xgboost_classifier.json  lambdamart_model.json

# 5. Predizione: uno score per ogni task
#scores = model.predict(X_val)
scores = model.predict_proba(X_val)[:, 1]

# 6. Associamo ogni score al rispettivo task e snapshot
results = pd.DataFrame({
    "sample_id": qid_val.to_numpy(),
    "row_index": row_indices,
    "score": scores
})

# 7. Ordiniamo i task per score decrescente all'interno di ogni snapshot
results = results.sort_values(
    ["sample_id", "score"],
    ascending=[True, False]
)

# 8. Selezioniamo i primi 3 task per ogni snapshot
top3 = results.groupby("sample_id").head(3)

# 9. Creiamo una riga per snapshot, con i 3 row_index separati da spazi
submission = (
    top3.groupby("sample_id")["row_index"]
    .apply(lambda x: " ".join(map(str, x)))
    .reset_index(name="prediction")
)

# 10. Salviamo il CSV
submission.to_csv("submission.csv", index=False)

print(f"Creato submission.csv con {len(submission)} snapshot")
print(submission.head())