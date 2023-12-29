
import pandas as pd
from sklearn import model_selection
from args import args

file_path = "/home/ajay/SchoolHack/FineTuning/data/corrected_df.csv"
try:
    corrected_df = pd.read_csv(file_path)
    print(corrected_df["context"].value_counts())
    # print(corrected_df['query_eng'].isna().sum())
    corrected_df = corrected_df.dropna()
    corrected_df.reset_index(drop=True, inplace=True)
except pd.errors.ParserError:
    # Handle a possible buffer overflow by reading the file in chunks
    chunk_size = 10000  # Adjust the chunk size based on your file size and memory constraints
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
    corrected_df = pd.concat(chunks, ignore_index=True)
    corrected_df = corrected_df.dropna()
    corrected_df.reset_index(drop=True, inplace=True)
    print(corrected_df['query_eng'].isna().sum())
    print(corrected_df)
corrected_df["kfold"] = -1
y = corrected_df.context.values
kf = model_selection.StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
for f, (t_, v_) in enumerate(kf.split(X=corrected_df, y=y)):
    corrected_df.loc[v_, 'kfold'] = f

# save the new csv with kfold column
corrected_df.to_csv("train_folds.csv", index=False)