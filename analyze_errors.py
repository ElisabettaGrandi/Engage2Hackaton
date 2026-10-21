import pandas as pd
from xgboost import XGBRanker
from preprocessing import preprocess_memory_efficient

# ============================================================
# 1. CARICAMENTO DATASET
# ============================================================

df = pd.read_csv(
    "data/engage-ktn-2026-competition/validation.csv"
)

# ============================================================
# 2. PREPROCESSING
# ============================================================

X, y, qid = preprocess_memory_efficient(df.copy())

# Il preprocessing ordina per sample_id.
# Applichiamo lo stesso ordinamento al DataFrame originale.
df = df.sort_values("sample_id").reset_index(drop=True)

assert len(df) == len(X)
assert df["sample_id"].reset_index(drop=True).equals(
    qid.reset_index(drop=True)
)

# ============================================================
# 3. CARICAMENTO MODELLO E PREDIZIONI
# ============================================================

model = XGBRanker()
model.load_model("lambdamart_model.json")

scores = model.predict(X)

assert len(scores) == len(df)

df["score"] = scores

# ============================================================
# 4. RANKING DEI TASK
# ============================================================

# Stessa logica utilizzata in validation.py:
# ordinamento per sample_id e score decrescente.

df = df.sort_values(
    ["sample_id", "score"],
    ascending=[True, False],
    kind="stable"
).reset_index(drop=True)

# Assegniamo una posizione a ciascun task nello snapshot.
df["predicted_rank"] = (
    df.groupby("sample_id").cumcount() + 1
)

# ============================================================
# 5. IDENTIFICAZIONE DEL TASK CORRETTO
# ============================================================

correct_tasks = df[df["target_index"] == 1].copy()

# Verifichiamo che ci sia un solo task corretto per snapshot.
assert len(correct_tasks) == df["sample_id"].nunique()

correct_tasks["hit"] = (
    correct_tasks["predicted_rank"] <= 3
)

# Numero di task presenti nello snapshot.
snapshot_sizes = df.groupby("sample_id").size()

correct_tasks["n_tasks"] = (
    correct_tasks["sample_id"].map(snapshot_sizes)
)

# ============================================================
# 6. RISULTATO GENERALE
# ============================================================

accuracy = correct_tasks["hit"].mean()

print("\n=== RISULTATO GENERALE ===")
print(f"Top-3 Hit Rate: {accuracy:.2%}")
print(f"Snapshot analizzati: {len(correct_tasks)}")
print(f"Snapshot corretti: {correct_tasks['hit'].sum()}")
print(f"Snapshot sbagliati: {(~correct_tasks['hit']).sum()}")

# ============================================================
# 7. POSIZIONE DEL TASK CORRETTO
# ============================================================

print("\n=== POSIZIONE DEL TASK CORRETTO ===")

print(
    correct_tasks["predicted_rank"]
    .value_counts()
    .sort_index()
    .head(15)
)

# ============================================================
# 8. ACCURACY PER TIPOLOGIA DI TASK
# ============================================================

print("\n=== ACCURACY PER TIPOLOGIA DI TASK ===")

task_analysis = (
    correct_tasks.groupby("task_index", observed=True)["hit"]
    .agg(["mean", "count"])
    .sort_values("mean")
)

task_analysis["mean"] *= 100

print(task_analysis)

# ============================================================
# 9. ACCURACY PER SESSIONE
# ============================================================

print("\n=== ACCURACY PER SESSIONE ===")

session_analysis = (
    correct_tasks.groupby("session")["hit"]
    .agg(["mean", "count"])
    .sort_values("mean")
)

session_analysis["mean"] *= 100

print(session_analysis)

# ============================================================
# 10. ACCURACY PER NUMERO DI TASK
# ============================================================

print("\n=== ACCURACY PER NUMERO DI TASK ===")

correct_tasks["size_group"] = pd.cut(
    correct_tasks["n_tasks"],
    bins=[0, 10, 20, 30, 40, 50, float("inf")]
)

size_analysis = (
    correct_tasks.groupby("size_group", observed=True)["hit"]
    .agg(["mean", "count"])
)

size_analysis["mean"] *= 100

print(size_analysis)

# ============================================================
# 11. SALVATAGGIO ERRORI
# ============================================================

errors = correct_tasks[~correct_tasks["hit"]].copy()

errors.to_csv(
    "validation_errors.csv",
    index=False
)

print("\n=== ANALISI COMPLETATA ===")
print(f"Errori salvati in validation_errors.csv: {len(errors)}")

# ============================================================
# 12. VERIFICA CON LO SCORER UFFICIALE
# ============================================================

# Generiamo la submission direttamente dalle predizioni
# appena calcolate, per verificare il risultato.

top3 = df[df["predicted_rank"] <= 3]

submission = (
    top3.groupby("sample_id")["row_index"]
    .apply(lambda x: " ".join(map(str, x)))
    .reset_index(name="prediction")
)

submission.to_csv(
    "submission_analysis.csv",
    index=False
)

print("\nSubmission di verifica salvata in submission_analysis.csv")