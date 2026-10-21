import pandas as pd
from xgboost import XGBRanker
from preprocessing import preprocess_test

# Caricamento del test
df = pd.read_csv("data/engage-ktn-2026-competition/test.csv")

# Preprocessing
X_test, qid_test, row_indices = preprocess_test(df)


# Caricamento del modello addestrato
model = XGBRanker()
model.load_model("lambdamart_model.json")

# Predizione
scores = model.predict(X_test)

# Associazione degli score ai task
results = pd.DataFrame({
    "sample_id": qid_test.to_numpy(),
    "row_index": row_indices,
    "score": scores
})

# Ordinamento dei task per priorità
results = results.sort_values(
    ["sample_id", "score"],
    ascending=[True, False]
)

# Selezione dei tre task migliori per snapshot
top3 = results.groupby("sample_id").head(3)

# Creazione del CSV
submission = (
    top3.groupby("sample_id")["row_index"]
    .apply(lambda x: " ".join(map(str, x)))
    .reset_index(name="prediction")
)

# Controllo rispetto al file ufficiale
sample = pd.read_csv(
    "data/engage-ktn-2026-competition/sample_submission.csv"
)

assert submission.columns.tolist() == ["sample_id", "prediction"]
assert set(submission["sample_id"]) == set(sample["sample_id"])
assert submission["prediction"].apply(
    lambda x: len(x.split()) == 3 and len(set(x.split())) == 3
).all()

# Salvataggio
submission.to_csv("submission_test.csv", index=False)

print(f"Submission generata: {len(submission)} snapshot")
print(submission.head())