import gzip
import json
import pandas as pd
import re
import os

def parse_price(price_val):
    if price_val is None:
        return None
    if isinstance(price_val, (int, float)):
        return float(price_val)
    if isinstance(price_val, str):
        match = re.search(r'(\d+\.\d+|\d+)', price_val)
        if match:
            return float(match.group(1))
    return None

def extract_subcategory(title):
    title = title.lower()
    categories = {
        'Dress': ['dress', 'gown', 'maxi', 'midi'],
        'Shirt/Top': ['shirt', 't-shirt', 'top', 'blouse', 'tee', 'tank'],
        'Pants/Bottom': ['pants', 'shorts', 'leggings', 'jeans', 'trousers', 'skirt'],
        'Shoes': ['shoes', 'sneakers', 'boots', 'sandals', 'heels', 'flats'],
        'Jewelry': ['jewelry', 'necklace', 'earring', 'bracelet', 'ring', 'pendant', 'locket'],
        'Underwear/Bra': ['bra', 'underwear', 'briefs', 'panties', 'lingerie', 'pantie'],
        'Outerwear': ['jacket', 'coat', 'hoodie', 'sweater', 'cardigan', 'blazer'],
        'Accessories': ['bag', 'watch', 'belt', 'sunglasses', 'scarf', 'hat', 'cap', 'wallet'],
        'Socks': ['sock', 'socks', 'hosiery'],
        'Swimwear': ['swimsuit', 'bikini', 'swim'],
        'Costume': ['costume', 'cosplay']
    }
    for cat, keywords in categories.items():
        if any(kw in title for kw in keywords):
            return cat
    return 'Other Clothing'

def prepare_dataset():
    input_file = "joined_reviews.jsonl.gz"
    output_file = "fashion_train_subset.csv"
    vote_threshold = 5
    
    data_list = []
    
    print(f"Reading {input_file} and finalizing dataset (Threshold >= {vote_threshold})...")
    
    count = 0
    with gzip.open(input_file, 'rt', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            
            votes = obj.get('helpful_vote', 0)
            if votes < vote_threshold:
                continue
                
            p_info = obj.get('product_info')
            if not p_info:
                continue
            
            p_title = p_info.get('title', '')
            p_desc = " ".join(p_info.get('description', []))
            p_features = " ".join(p_info.get('features', []))
            
            # Combine all text context for the LLM Branch
            # [Product Name] [Review Title] [Review Text] [Product Specs]
            combined_context = f"Product: {p_title} | Review Title: {obj.get('title', '')} | Review: {obj.get('text', '')} | Specs: {p_desc} {p_features}".strip()
            
            row = {
                # Target
                'target': obj.get('rating'),
                
                # Text Branch (LLaMA input)
                'input_text': combined_context,
                
                # Tabular Branch (MLP input)
                'brand': p_info.get('store', 'Unknown'),
                'sub_category': extract_subcategory(p_title),
                'prod_avg_rating': p_info.get('average_rating'),
                'prod_rating_count': p_info.get('rating_number', 0),
                'price_raw': p_info.get('price'),
                
                # Identifiers
                'parent_asin': obj.get('parent_asin')
            }
            
            data_list.append(row)
            count += 1
            if count % 10000 == 0:
                print(f"  Processed {count} rows...")

    print(f"Extraction complete. Total rows: {len(data_list)}")
    
    df = pd.DataFrame(data_list)
    
    # Clean Price
    df['price_clean'] = df['price_raw'].apply(parse_price)
    df['price_missing'] = df['price_clean'].isna().astype(int)
    median_price = df['price_clean'].median()
    df['price_clean'] = df['price_clean'].fillna(median_price if not pd.isna(median_price) else 0.0)
    
    df = df.dropna(subset=['target'])
    
    # Save final cleaned version
    final_cols = [
        'target', 'input_text', 'brand', 'sub_category', 
        'prod_avg_rating', 'prod_rating_count', 'price_clean', 
        'price_missing', 'parent_asin'
    ]
    df = df[final_cols]
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("Success! High-quality dataset is ready.")

if __name__ == '__main__':
    prepare_dataset()
