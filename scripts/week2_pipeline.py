import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.text_cleaning import TextCleaner

def run_pipeline():
    print("=== Week 2: Text Cleaning & Normalization Pipeline ===")
    
    # 1. Load data
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'sample_listing.csv'))
    output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cleaned_listing.csv'))
    
    print(f"\n[1] Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} listings.")
    
    # 2. Profiling Report before cleaning
    cleaner = TextCleaner()
    print("\n[2] Generating Data Profiling Report (Before Cleaning)...")
    profile = cleaner.profile_column(df, 'remarks')
    
    print(f" - Null Rate: {profile.get('null_rate', 0):.2%}")
    print(f" - Average Remark Length: {profile.get('avg_length', 0):.1f} characters")
    print(f" - Price Mentions (contains $x): {profile.get('price_mentions', 0)}")
    print(f" - HTML Tags Found In: {profile.get('has_html', 0)} listings")
    print(f" - Common Terms: {profile.get('common_terms', {})}")
    print(f" - Common Abbreviations Found: {profile.get('common_abbreviations', {})}")
    
    # 3. Apply cleaning
    print("\n[3] Applying TextCleaner...")
    df['cleaned_remarks'] = df['remarks'].apply(lambda x: cleaner.clean_text(str(x)) if pd.notnull(x) else "")
    
    # 4. Save cleaned dataset
    print(f"\n[4] Saving cleaned dataset to: {output_file}")
    df.to_csv(output_file, index=False)
    
    # 5. Before/After Examples
    print("\n[5] Generating Before/After Examples:")
    # Get a sample of 10 rows where the text was actually changed
    changed_df = df[df['remarks'].astype(str) != df['cleaned_remarks']]
    sample_size = min(10, len(changed_df))
    
    if sample_size > 0:
        sample_df = changed_df.sample(sample_size, random_state=42)
        for i, (_, row) in enumerate(sample_df.iterrows(), 1):
            print(f"\nExample {i}:")
            # Truncate to keep output manageable
            original = str(row['remarks'])[:150] + "..." if len(str(row['remarks'])) > 150 else str(row['remarks'])
            cleaned = str(row['cleaned_remarks'])[:150] + "..." if len(str(row['cleaned_remarks'])) > 150 else str(row['cleaned_remarks'])
            print(f"BEFORE: {original}")
            print(f"AFTER:  {cleaned}")
    else:
        print("No changes were made to any text.")
        
    print("\n=== Pipeline Complete ===")

if __name__ == '__main__':
    run_pipeline()
