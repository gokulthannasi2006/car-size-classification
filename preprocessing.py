import pandas as pd

df = pd.read_csv("cars(2).csv")
df.columns = df.columns.str.strip()
df = df.dropna(axis=1, how="all").drop_duplicates()
target = "Size"
drops = [c for c in ["TYPE", "Unnamed: 5", "(g/km)", "RATING", "Model"] if c in df.columns]
df = df.drop(columns=drops).dropna(subset=[target])
for c in df.columns:
    if c != target:
        conv = pd.to_numeric(df[c], errors="coerce")
        if df[c].notna().sum() and conv.notna().sum() >= .7 * df[c].notna().sum(): df[c] = conv
for c in df.columns:
    if c == target: continue
    if pd.api.types.is_numeric_dtype(df[c]): df[c] = df[c].fillna(df[c].median())
    else:
        m = df[c].mode(dropna=True); df[c] = df[c].fillna(m.iloc[0] if len(m) else "Unknown")
X = df.drop(columns=[target]); y = df[target].astype(str)
X = pd.get_dummies(X, columns=X.select_dtypes(include="object").columns, drop_first=True, dtype=int)
pd.concat([X, y.rename(target)], axis=1).to_csv("cars_size_classification_preprocessed.csv", index=False)
print("Preprocessing completed!")
print(y.value_counts())
