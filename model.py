import pandas as pd
from xgboost import XGBRanker
from preprocessing import preprocess_memory_efficient


train_df = pd.read_csv(
    "data/engage-ktn-2026-competition/train.csv"
)

X_train, y_train, qid_train = preprocess_memory_efficient(train_df)

# 1. Configurazione del modello
model = XGBRanker(
    enable_categorical=True,
    objective="rank:ndcg",
    eval_metric="ndcg@3",
    tree_method="hist",#hist
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    lambdarank_pair_method="topk",
    lambdarank_num_pair_per_sample=6, #5 o 4
    random_state=42
)

# 2. Addestramento
model.fit(
    X_train,
    y_train,
    qid=qid_train
)

# 3. Salvataggio del modello
model.save_model("lambdamart_model.json")