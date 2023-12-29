import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os 


import json
import pandas as pd
file_path = "/home/ajay/SchoolHack/FineTuning/data/2023-10-01_2023-10-15-cleaned.json"
with open(file_path, 'r') as file:
 data = json.load(file)
extracted_data = []

for record in data:
 # print(record)
 for session in data['users']:
     # print(session)
     for message in session['sessions']:
         # print (message)
         for message_ in message['messages']:
             if 'metadata' in message_ and 'context' in message_['metadata']:
                 context = message_['metadata']['context']
                 if context and context != None:
                     extracted_data.append({
                         'query_eng': message_.get('query_eng', ''),
                         'complexity': message_['metadata'].get('complexity', ''),
                         'context': context
                     })

# df = pd.DataFrame(extracted_data)
# df['context'] = df['context'].apply(str)

# # Now you can find unique values
# unique_classes = df['context'].unique()
# df['complexity'] = df['complexity'].apply(str)

# # Now you can find unique values
# unique_classes_comlexity = df['complexity'].unique()
# unique_classes_comlexity


# selected_values = ["scientific", "historical", "mathematical", "biology", "health", "medical", "social", "political", "technical", "legal", "artistic", "philosophical", "biological", "healthcare"]

# new_values = ["social science", "history", "social sciences", "mathematics", "unidentified", "unknown", "none", "personal"]

# selected_values.extend(new_values)


# df_updated_context = df[df['context'].isin(selected_values)]


# df_updated_context['context'] = df_updated_context['context'].replace({'biological': 'biology', 'historical': 'history', 'healthcare':'health'})
# df_updated_context['context'] = df_updated_context['context'].replace({'mathematical': 'mathematics'})
# values_to_replace = ['unidentified', 'unknown', 'none', 'personal']
# df_updated_context['context'] = df_updated_context['context'].replace(values_to_replace, 'other')
# df_updated_context.reset_index(drop= True, inplace= True)
# df_updated_context

# df_updated_context['context'].value_counts()

# df_updated_context = df_updated_context[df_updated_context["context"] != "social sciences"]
# df_updated_context = df_updated_context[df_updated_context["context"] != "social science"]
# df_updated_context.reset_index(drop = True, inplace = True)

# df_updated_context = df_updated_context.sample(frac = 1)

# df_updated_context.reset_index(drop = True, inplace = True)


# corrected_df = df_updated_context
# corrected_df["context"].value_counts()

# corrected_df.to_csv("./data/corrected_df.csv", index = False)
                     
df = pd.DataFrame(extracted_data)
df['context'] = df['context'].astype(str)
df['complexity'] = df['complexity'].astype(str)

selected_values = ["scientific", "historical", "mathematical", "biology", "health", "medical", "social", "political", "technical", "legal", "artistic", "philosophical", "biological", "healthcare"]
new_values = ["social science", "history", "social sciences", "mathematics", "unidentified", "unknown", "none", "personal"]
selected_values.extend(new_values)

df_updated_context = df[df['context'].isin(selected_values)].copy()
context_replacement = {
    'biological': 'biology', 
    'historical': 'history', 
    'healthcare':'health', 
    'mathematical': 'mathematics'
}
df_updated_context.loc[:, 'context'] = df_updated_context['context'].replace(context_replacement)

values_to_replace = ['unidentified', 'unknown', 'none', 'personal']
df_updated_context.loc[:, 'context'] = df_updated_context['context'].replace(values_to_replace, 'other')
df_updated_context = df_updated_context[~df_updated_context["context"].isin(["social sciences", "social science"])]
df_updated_context = df_updated_context.sample(frac=1).reset_index(drop=True)

corrected_df = df_updated_context
corrected_df = corrected_df.dropna(subset = ["query_eng"])
corrected_df.reset_index(drop = True, inplace = True)

print(corrected_df['query_eng'].isna().sum())
corrected_df.to_csv("./data/corrected_df.csv", index=False)

print(corrected_df)

# print(corrected_df["context"].value_counts())

file_path = "/home/ajay/SchoolHack/FineTuning/data/corrected_df.csv"
try:
    corrected_df = pd.read_csv(file_path)
    print(corrected_df["context"].value_counts())
    print(corrected_df['query_eng'].isna().sum())
except pd.errors.ParserError:
    # Handle a possible buffer overflow by reading the file in chunks
    chunk_size = 10000  # Adjust the chunk size based on your file size and memory constraints
    chunks = pd.read_csv(file_path, chunksize=chunk_size)
nan_count = corrected_df['query_eng'].isna().sum()

print(corrected_df.describe())

print(nan_count)