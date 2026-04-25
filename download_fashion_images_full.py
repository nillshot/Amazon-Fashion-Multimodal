import pandas as pd
import json
import os
import requests
import concurrent.futures
from tqdm import tqdm

# 설정
CSV_PATH = 'fashion_train_subset_2.csv'
META_PATH = os.path.join('meta_Amazon_Fashion.jsonl', 'meta_Amazon_Fashion.jsonl')
IMAGE_DIR = 'images'
OUTPUT_CSV = 'fashion_train_subset_2_with_images.csv'

def download_image(args):
    asin, url, save_path = args
    if not url:
        return asin, False, None
    
    # 이미 다운로드 된 파일이 있으면 건너뛰기
    if os.path.exists(save_path):
        return asin, True, save_path
        
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return asin, True, save_path
    except Exception as e:
        # print(f"Error downloading {asin}: {e}")
        return asin, False, None

def main():
    print("1. 데이터 로드 및 대상 ASIN 추출")
    df = pd.read_csv(CSV_PATH)
    
    # 전체 고유한 parent_asin 추출
    target_asins = df['parent_asin'].dropna().unique()
    target_asin_set = set(target_asins)
    print(f"추출된 타겟 ASIN 개수: {len(target_asin_set)}")

    # 이미지 저장 폴더 생성
    os.makedirs(IMAGE_DIR, exist_ok=True)

    print("2. 메타데이터에서 이미지 URL 검색")
    asin_to_url = {}
    with open(META_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                asin = data.get('parent_asin')
                if asin in target_asin_set:
                    # 'images' 속성에서 고해상도 이미지 또는 기본 큰 이미지 URL 찾기
                    images = data.get('images', [])
                    url = None
                    if images:
                        primary_img = images[0]
                        url = primary_img.get('hi_res') or primary_img.get('large')
                    
                    if url:
                        asin_to_url[asin] = url
            except json.JSONDecodeError:
                continue

    print(f"URL을 찾은 ASIN 개수: {len(asin_to_url)}")

    print("3. 이미지 다운로드 진행")
    download_tasks = []
    for asin, url in asin_to_url.items():
        ext = url.split('.')[-1]
        save_path = os.path.join(IMAGE_DIR, f"{asin}.{ext}")
        download_tasks.append((asin, url, save_path))

    asin_to_local_path = {}
    # 멀티스레딩으로 빠른 다운로드 처리 (워커 수 증가)
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(download_image, download_tasks), total=len(download_tasks), desc="Downloading"))

    success_count = 0
    for asin, success, path in results:
        if success:
            success_count += 1
            asin_to_local_path[asin] = path
            
    print(f"다운로드 성공/이미 존재: {success_count} / {len(download_tasks)}")

    print("4. 최종 데이터셋 생성")
    # 다운로드 성공한 로컬 이미지 경로 매핑
    df['image_path'] = df['parent_asin'].map(asin_to_local_path)
    
    # 다운로드 실패한 데이터 제외 (선택사항)
    filtered_df = df.dropna(subset=['image_path'])
    
    # 저장
    filtered_df.to_csv(OUTPUT_CSV, index=False)
    print(f"작업 완료! 결과가 {OUTPUT_CSV}에 저장되었습니다. (총 데이터: {len(filtered_df)}건)")

if __name__ == "__main__":
    main()
