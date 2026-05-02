import pandas as pd
import gzip
import json
import re
from tqdm import tqdm
import os

def add_helpful_votes():
    csv_path = 'fashion_train_subset_2_with_images.csv'
    origin_data_path = 'joined_reviews.jsonl.gz'
    output_path = 'fashion_train_subset_3_with_help.csv'

    if not os.path.exists(csv_path) or not os.path.exists(origin_data_path):
        print(f"Error: Required files not found in the current directory.")
        return

    # 1. Load the current CSV
    print("1. 현재 CSV 파일 불러오는 중...")
    df = pd.read_csv(csv_path)

    # 2. Extract (parent_asin, review_text) -> helpful_vote mapping
    help_map = {}
    print("2. 원본 데이터(jsonl.gz)에서 매핑 생성 중... (파일이 커서 시간이 조금 걸릴 수 있습니다)")
    with gzip.open(origin_data_path, 'rt', encoding='utf-8') as f:
        # Since the file is 1.1GB, we'll process it efficiently
        for line in tqdm(f, desc="Processing original records"):
            try:
                obj = json.loads(line)
                asin = obj.get('parent_asin')
                text = obj.get('text', '').strip()
                votes = obj.get('helpful_vote', 0)
                
                if asin and text:
                    # Keep the highest vote if there are duplicates (unlikely, but safe)
                    key = (asin, text)
                    if key not in help_map or votes > help_map[key]:
                        help_map[key] = votes
            except Exception:
                continue

    # 3. Match and Add Column
    print("3. 현재 데이터에 도움 수(helpful_vote) 매칭 중...")
    
    def find_help_vote(row):
        input_text = str(row['input_text'])
        # Extract the review text from the combined input_text format
        match = re.search(r"Review: (.*?) \| Specs:", input_text)
        if match:
            review_text = match.group(1).strip()
            # Lookup in the map
            return help_map.get((row['parent_asin'], review_text), 0)
        return 0

    tqdm.pandas(desc="Matching votes")
    df['helpful_vote'] = df.progress_apply(find_help_vote, axis=1)

    # Calculate match statistics
    matched_count = (df['helpful_vote'] > 0).sum()
    print(f"  - 총 {len(df)}개 중 {matched_count}개 리뷰의 도움 수 매칭 완료.")
    
    if matched_count == 0:
         print("  - 주의: 매칭된 리뷰가 없습니다. 매칭 로직을 확인해야 할 수 있습니다.")

    # 4. Save the result
    print(f"4. 결과를 {output_path} 로 저장하는 중...")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("✅ 완료! 가중치 학습에 사용할 수 있는 새로운 CSV 파일이 생성되었습니다.")

if __name__ == "__main__":
    add_helpful_votes()
