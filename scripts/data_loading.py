import csv
import random
import os
import ast

def extract_sample_from_sql(sql_file, output_file, sample_size=1000):
    import re
    print(f"Reading SQL file: {sql_file}")
    
    remarks = []
    
    print("Extracting remarks...")
    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        in_insert = False
        for line in f:
            if line.startswith('INSERT INTO `rets_property`'):
                in_insert = True
                
            if in_insert:
                # Find any string enclosed in single quotes that is at least 60 characters
                matches = re.findall(r"'([^']{60,})'", line)
                for m in matches:
                    if 'http' not in m and '<' not in m and len(m) > 60:
                        remarks.append(m)
                
                if line.rstrip().endswith(';'):
                    in_insert = False

    print(f"Found {len(remarks)} total text fields resembling remarks.")
    
    # Clean and filter
    valid_remarks = list(set(remarks)) # remove duplicates
    
    print(f"Filtered down to {len(valid_remarks)} valid unique remarks.")
    
    # Sample
    actual_sample_size = min(sample_size, len(valid_remarks))
    if actual_sample_size == 0:
        print("No valid remarks found to sample.")
        return
        
    sampled_remarks = random.sample(valid_remarks, actual_sample_size)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Writing {actual_sample_size} remarks to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['remarks'])
        for r in sampled_remarks:
            writer.writerow([r])
            
    print("Done!")

def extract_from_values_string(text, remarks_list):
    pass


if __name__ == '__main__':
    extract_sample_from_sql('data/raw/rets_property.sql', 'data/processed/sample_listing.csv')
