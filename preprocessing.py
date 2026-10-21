import pandas as pd
import numpy as np

def preprocess_memory_efficient(df):

    # 1. Conversione di task_index in categoria
    df['task_index'] = df['task_index'].astype('category')

    # 2. Riduzione della memoria
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # 3. Riduzione della memoria per gli interi
    int_cols = df.select_dtypes(include=['int64']).columns

    for c in int_cols:
        if c not in ['sample_id', 'row_index']:
            df[c] = df[c].astype(np.int32)

    # 4. Ordinamento per snapshot
    df = df.sort_values('sample_id').reset_index(drop=True)

    # 5. Separazione delle label e degli identificativi
    y_train = df['target_index']
    qid_train = df['sample_id']

    # 6. Eliminazione delle colonne non utilizzate
    excluded_cols = [
        'sample_id',
        'row_index',
        'target_index',
        'session',
        'atco',
        'timestamp',
        'task_name'
    ]

    X_train = df.drop(columns=excluded_cols)

    return X_train, y_train, qid_train


def preprocess_test(df):
    df = df.copy()

    df["task_index"] = df["task_index"].astype("category")

    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for c in int_cols:
        if c not in ["sample_id", "row_index"]:
            df[c] = df[c].astype("int32")

    df = df.sort_values("sample_id").reset_index(drop=True)

    qid_test = df["sample_id"]
    row_indices = df["row_index"]

    excluded_cols = [
        "sample_id", "row_index", "target_index",
        "session", "atco",  "task_name", "aircraft", "aircraft_id"
    ]

    #"timestamp",

    X_test = df.drop(columns=excluded_cols, errors="ignore")

    return X_test, qid_test, row_indices