import pandas as pd
from xgboost import XGBClassifier
from preprocessing import preprocess_memory_efficient

train_df = pd.read_csv(
    "data/engage-ktn-2026-competition/train.csv"
)

X_train, y_train, qid_train = preprocess_memory_efficient(train_df)

model = XGBClassifier(
    enable_categorical=True,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    random_state=42
)

model.fit(X_train, y_train)

model.save_model("xgboost_classifier.json")