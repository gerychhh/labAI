
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

def detect_target(train_df: pd.DataFrame, test_df: pd.DataFrame):
    diff = [c for c in train_df.columns if c not in test_df.columns]
    if len(diff) == 1:
        return diff[0]
    for c in ["Transported", "Survived", "target", "label", "y"]:
        if c in train_df.columns and c not in test_df.columns:
            return c
    return None

def preprocess_basic(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "Cabin" in df.columns:
        parts = df["Cabin"].astype("string").str.split("/", expand=True)
        df["CabinDeck"] = parts[0]
        df["CabinSide"] = parts[2]
        df = df.drop(columns=["Cabin"])

    # убираем почти уникальные колонки (иначе OHE взорвётся)
    for col in ["PassengerId", "Name"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def main():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    # вывод данных
    print("=== TRAIN head ===")
    print(train_df.head())
    print("TRAIN shape:", train_df.shape)

    print("\n=== TEST head ===")
    print(test_df.head())
    print("TEST shape:", test_df.shape)

    #пропуски
    print("\n=== NaN count (TRAIN) ===")
    print(train_df.isna().sum())
    print("\n=== NaN count (TEST) ===")
    print(test_df.isna().sum())

    # target
    target_col = detect_target(train_df, test_df)
    print("\nDetected target:", target_col)

    y = None
    X = train_df.copy()
    if target_col is not None:
        y = X[target_col]
        X = X.drop(columns=[target_col])

    #базовая обработка 
    X = preprocess_basic(X)
    test_df = preprocess_basic(test_df)

    # split 70/30
    X_train = X.sample(frac=0.7, random_state=42)
    X_valid = X.drop(index=X_train.index)

    num_cols = X_train.select_dtypes(include="number").columns.tolist()
    cat_cols = X_train.select_dtypes(exclude="number").columns.tolist()

    # заполнение пропусков
    mean_col = "Age" if "Age" in num_cols else (num_cols[0] if num_cols else None)

    fill_num = {}
    for c in num_cols:
        if c == mean_col:
            fill_num[c] = ("mean", float(X_train[c].mean()))
        else:
            fill_num[c] = ("median", float(X_train[c].median()))

    fill_cat = {}
    for c in cat_cols:
        m = X_train[c].mode(dropna=True)
        fill_cat[c] = ("mode", m.iloc[0] if len(m) else "Unknown")

    def fill_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c, (how, v) in fill_num.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)
        for c, (how, v) in fill_cat.items():
            if c in df.columns:
                df[c] = df[c].fillna(v)
        return df.infer_objects(copy=False)

    X_train_f = fill_df(X_train)
    X_valid_f = fill_df(X_valid)
    test_f    = fill_df(test_df)

    print("\n=== Чем заполняли (по X_train) ===")
    for c, (how, v) in fill_num.items():
        print(f"NUM {c}: {how} = {v}")
    for c, (how, v) in fill_cat.items():
        print(f"CAT {c}: {how} = {v}")

    # показать что пропуски реально ушли
    print("\n=== NaN total AFTER filling ===")
    print("X_train:", int(X_train_f.isna().sum().sum()))
    print("X_valid:", int(X_valid_f.isna().sum().sum()))
    print("test   :", int(test_f.isna().sum().sum()))

    # чтоб новые категории не ломали структуру
    X_train_cat = pd.get_dummies(X_train_f[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=X_train_f.index)
    X_valid_cat = pd.get_dummies(X_valid_f[cat_cols], drop_first=True) if cat_cols else pd.DataFrame(index=X_valid_f.index)
    test_cat    = pd.get_dummies(test_f[cat_cols],    drop_first=True) if cat_cols else pd.DataFrame(index=test_f.index)

    X_valid_cat = X_valid_cat.reindex(columns=X_train_cat.columns, fill_value=0)
    test_cat    = test_cat.reindex(columns=X_train_cat.columns, fill_value=0)

    #нормализация по train
    means = X_train_f[num_cols].mean() if num_cols else pd.Series(dtype=float)
    stds  = X_train_f[num_cols].std(ddof=0).replace(0, 1) if num_cols else pd.Series(dtype=float)

    def norm_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not cols:
            return pd.DataFrame(index=df.index)
        return (df[cols] - means[cols]) / stds[cols]

    X_train_num = norm_num(X_train_f, num_cols)
    X_valid_num = norm_num(X_valid_f, num_cols)

    test_num_cols = [c for c in num_cols if c in test_f.columns]
    test_num = norm_num(test_f, test_num_cols)

    # финальные таблицы
    train_proc = pd.concat([X_train_num, X_train_cat], axis=1)
    valid_proc = pd.concat([X_valid_num, X_valid_cat], axis=1)
    test_proc  = pd.concat([test_num,    test_cat],    axis=1)

    if target_col is not None:
        train_proc[target_col] = y.loc[X_train.index].values
        valid_proc[target_col] = y.loc[X_valid.index].values

    print("\n=== Processed TRAIN head ===")
    print(train_proc.head())
    print("processed_train shape:", train_proc.shape)
    print("processed_valid shape:", valid_proc.shape)
    print("processed_test  shape:", test_proc.shape)

    train_proc.to_csv("processed_train.csv", index=False)
    valid_proc.to_csv("processed_valid.csv", index=False)
    test_proc.to_csv("processed_test.csv", index=False)
    print("\nSaved: processed_train.csv, processed_valid.csv, processed_test.csv")

if __name__ == "__main__":
    main()