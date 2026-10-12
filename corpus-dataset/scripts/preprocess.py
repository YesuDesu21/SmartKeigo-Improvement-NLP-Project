from datasets import load_dataset
import pandas as pd

# Data Ingestion

ds = load_dataset("ronantakizawa/japanese-honorifics")

data_0 = ds['train'].to_pandas() # Transform into df
data_1 = pd.read_csv("../raw/annot_data_1.csv")
data_2 = pd.read_csv("../raw/annot_data_2.csv")

total_rows = len(data_1) + len(data_2) + len(data_0)
print(f"Shape: {total_rows}")

# Data Integration

def standardize_label(label):
    if pd.isna(label):
        return 'unknown'
    
    # Standard string replacements
    s = str(label)
    
    # Prefix translations
    s = s.replace('変換:', '')
    s = s.replace('どちらでも良い:', 'optional_')
    s = s.replace('無変換', 'no_change')
    
    # Keigo type translations
    s = s.replace('丁寧語', 'teineigo')
    s = s.replace('尊敬語', 'sonkeigo')
    s = s.replace('謙譲語', 'kenjogo')
    
    # Separator translations
    s = s.replace('、', '_and_')
    s = s.replace('Um', 'unclear')
    
    return s.strip()

# 1. Unpivot data_0 into long format (1 row -> 3 rows)
data_0_long = pd.melt(
    data_0,
    id_vars=['base_sentence'],
    value_vars=['teineigo', 'sonkeigo', 'kenjogo'],
    var_name='transformation_label',
    value_name='transformed_text'
)

# Standardize base_sentence column to source_text
data_0_long = data_0_long.rename(columns={'base_sentence': 'source_text'})
data_0_long = data_0_long[['source_text', 'transformed_text', 'transformation_label']]
data_0_long = data_0_long.dropna(subset=['transformed_text'])

# 2. Rename Japanese columns in data_1 and data_2 to English
column_map = {
    '元文': 'source_text',
    '変換文': 'transformed_text',
    '変換ラベル': 'transformation_label'
}

data_1_en = data_1.rename(columns=column_map)
data_2_en = data_2.rename(columns=column_map)

# 3. Convert Japanese keigo labels in data_1 and data_2 into Romaji / English
label_map = {
    '変換:丁寧語': 'teineigo',
    '変換:尊敬語': 'sonkeigo',
    '変換:謙譲語': 'kenjogo',
    '無変換': 'no_change'
}

data_1_en['transformation_label'] = data_1_en['transformation_label'].apply(standardize_label)
data_2_en['transformation_label'] = data_2_en['transformation_label'].apply(standardize_label)

# 4. Combine all datasets and drop exact duplicates
df_combined = pd.concat([data_0_long, data_1_en, data_2_en], ignore_index=True)
df_combined = df_combined.drop_duplicates().reset_index(drop=True)

df_combined.to_csv('../processed/integrated_csv.csv', index=False)