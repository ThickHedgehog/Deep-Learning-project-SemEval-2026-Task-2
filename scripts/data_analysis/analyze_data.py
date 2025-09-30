import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent.parent

# Load the dataset
df = pd.read_csv(project_root / 'data/raw/train_subtask1.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("=== DATASET ANALYSIS ===")
print(f"Dataset shape: {df.shape}")
print(f"Unique users: {df['user_id'].nunique()}")
print(f"Columns: {list(df.columns)}")
print(f"Collection phases: {sorted(df['collection_phase'].unique())}")
print(f"Text types (is_words): {df['is_words'].value_counts().to_dict()}")
print(f"Valence range: {df['valence'].min()} to {df['valence'].max()}")
print(f"Arousal range: {df['arousal'].min()} to {df['arousal'].max()}")

print("\n=== SEQUENCE ANALYSIS FOR USER 3 ===")
user_3 = df[df['user_id'] == 3].sort_values('timestamp')
print(f"Total entries for user 3: {len(user_3)}")
print("First 5 entries:")
for _, row in user_3.head(5).iterrows():
    print(f"  text_id: {row['text_id']}, timestamp: {row['timestamp']}, "
          f"valence: {row['valence']}, arousal: {row['arousal']}")

print("\n=== USER STATISTICS ===")
user_stats = df.groupby('user_id').agg({
    'text_id': 'count',
    'timestamp': ['min', 'max'],
    'collection_phase': 'nunique'
}).round(2)
user_stats.columns = ['num_entries', 'first_timestamp', 'last_timestamp', 'num_phases']
print("Summary stats by user (first 10 users):")
print(user_stats.head(10))

print("\n=== TEXT TYPE ANALYSIS ===")
print(f"Essay entries (is_words=False): {(~df['is_words']).sum()}")
print(f"Word entries (is_words=True): {df['is_words'].sum()}")

# Sample texts
print("\n=== SAMPLE TEXTS ===")
print("Sample essay (is_words=False):")
essay_sample = df[~df['is_words']]['text'].iloc[0]
print(f"  {essay_sample[:200]}...")

print("\nSample word list (is_words=True):")
words_sample = df[df['is_words']]['text'].iloc[0]
print(f"  {words_sample}")