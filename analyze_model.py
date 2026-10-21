import pandas as pd
from xgboost import XGBRanker

model = XGBRanker()
model.load_model("lambdamart_model.json")

booster = model.get_booster()

importance = booster.get_score(importance_type="total_gain")

importance_df = pd.DataFrame(
    importance.items(),
    columns=["feature", "importance"]
)

importance_df = importance_df.sort_values(
    "importance", ascending=False
)

print(importance_df.to_string(index=False))