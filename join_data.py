import gzip
import json
import time
import os

def join_data():
    meta_file = "meta_Amazon_Fashion.jsonl.gz"
    review_file = "Amazon_Fashion.jsonl.gz"
    output_file = "joined_reviews.jsonl.gz"

    print("Phase 1: Loading Product Metadata into memory...")
    start_time = time.time()
    
    meta_dict = {}
    lines_read = 0
    with gzip.open(meta_file, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                # We load everything. To save space, we store only the parts we need,
                # but the user requested *all* info. 
                # To minimize memory overhead compared to deeply nested Python dicts, 
                # we store the raw JSON string instead of the dict. We will parse it on demand.
                # Actually, wait, let's just parse it, extract parent_asin, then del it from the obj to not duplicate.
                obj = json.loads(line)
                parent_asin = obj.get('parent_asin')
                if parent_asin:
                    del obj['parent_asin'] # avoid duplication
                    # Storing raw dict might take a lot of memory. To avoid memory overhead,
                    # we store the dumped string.
                    meta_dict[parent_asin] = json.dumps(obj)
            except Exception as e:
                print(f"Error parsing meta on line {lines_read}: {e}")
            
            lines_read += 1
            if lines_read % 100000 == 0:
                print(f"  Read {lines_read} metadata entries...")

    print(f"Phase 1 Complete! Loaded {len(meta_dict)} unique products. Time taken: {time.time() - start_time:.2f}s")
    
    print("\nPhase 2: Streaming Reviews and Joining Metadata...")
    start_time = time.time()
    
    reviews_processed = 0
    reviews_matched = 0
    with gzip.open(review_file, 'rt', encoding='utf-8') as f_in, \
         gzip.open(output_file, 'wt', encoding='utf-8') as f_out:
         
        for line in f_in:
            try:
                review_obj = json.loads(line)
                parent_asin = review_obj.get('parent_asin')
                
                # If we have metadata for this product
                if parent_asin in meta_dict:
                    # Parse the product info and attach it
                    product_info = json.loads(meta_dict[parent_asin])
                    review_obj['product_info'] = product_info
                    reviews_matched += 1
                else:
                    review_obj['product_info'] = None
                
                # Write to output file
                f_out.write(json.dumps(review_obj) + '\n')
                
            except Exception as e:
                print(f"Error parsing review on line {reviews_processed}: {e}")
                
            reviews_processed += 1
            if reviews_processed % 500000 == 0:
                print(f"  Processed {reviews_processed} reviews... (Matched: {reviews_matched})")
                
    print(f"Phase 2 Complete! Processed {reviews_processed} reviews. Time taken: {time.time() - start_time:.2f}s")
    print(f"Total matched reviews: {reviews_matched}")
    
    # Print file sizes
    meta_size = os.path.getsize(meta_file) / (1024*1024)
    rev_size = os.path.getsize(review_file) / (1024*1024)
    out_size = os.path.getsize(output_file) / (1024*1024)
    print(f"\nFinal Statistics:")
    print(f"  Metadata File: {meta_size:.2f} MB")
    print(f"  Input Review File: {rev_size:.2f} MB")
    print(f"  Output Joined File: {out_size:.2f} MB")
    print(f"  Output File Location: {os.path.abspath(output_file)}")

if __name__ == '__main__':
    join_data()
