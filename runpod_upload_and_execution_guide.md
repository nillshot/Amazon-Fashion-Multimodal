# 🚀 RunPod 업로드 및 모델 학습 실행 완벽 가이드

RunPod JupyterLab 화면에 성공적으로 접속하신 것을 축하드립니다! 현재 상태에서 **로컬 데이터와 사전학습 가중치를 RunPod 서버로 빠르고 안전하게 업로드하고, 학습을 실행하는 로드맵**입니다.

3.6GB 크기의 `images.zip`과 500MB 크기의 `pretrained_fashion_roberta.pth`는 **JupyterLab 웹 브라우저 업로드 버튼을 사용하면 속도가 매우 느리거나 중간에 끊길 위험**이 있습니다. 따라서 가장 효율적이고 빠른 방법들을 엄선하여 단계별로 안내해 드립니다.

---

## 🗺️ 1단계. RunPod 최종 폴더 구조 목표

업로드가 모두 완료되었을 때 RunPod의 `/workspace` 내부 구조는 다음과 같아야 합니다:

```text
/workspace/
├── fashion_train_subset_2_with_images.csv  # 학습용 CSV 데이터셋
├── pretrained_fashion_roberta.pth          # (선택) 텍스트 사전학습 가중치 (500MB)
├── pretrained_fashion_mobilenet.pth        # (선택) 이미지 사전학습 가중치 (12MB)
├── multimodal_mobile_version_v9_runpod.py  # 실행할 V9 최적화 학습 스크립트
└── images/                                 # 압축 해제된 상품 이미지 폴더
    ├── B07CCB3YG6.jpg
    ├── B07PGST3RL.jpg
    └── ...
```

---

## 📤 2단계. 데이터 업로드 방법 선택하기

본인의 네트워크 환경과 선호도에 맞춰 **방법 A** 또는 **방법 B** 중 하나를 선택하여 대용량 파일을 업로드하세요.

### 💡 방법 A. 구글 드라이브 & `gdown` 활용 (가장 추천! 속도 초당 50~100MB)
대용량 파일(`images.zip`, `pretrained_fashion_roberta.pth`)을 구글 드라이브에 올린 뒤, RunPod 터미널에서 명령어로 직접 다운로드하는 방법입니다. RunPod 데이터센터 대역폭을 사용하여 수 분 내에 다운로드가 끝납니다.

1. **로컬 파일 구글 드라이브 업로드**:
   * `images.zip`과 `pretrained_fashion_roberta.pth`를 구글 드라이브에 업로드합니다.
   * 각 파일의 링크를 생성하고 **"링크가 있는 모든 사용자에게 공개" (뷰어)**로 설정을 변경합니다.
   * 링크 주소에서 **파일 ID**를 추출합니다.
     * 예: `https://drive.google.com/file/d/1A2B3C4D_XYZ/view?usp=sharing` 에서 파일 ID는 `1A2B3C4D_XYZ` 입니다.

2. **JupyterLab 터미널 열기**:
   * 화면 중앙 Launcher에서 **`Terminal`**을 클릭하여 터미널을 엽니다.

3. **`gdown` 설치 및 다운로드**:
   * 터미널에 아래 명령어를 입력하여 다운로드 툴을 설치합니다:
     ```bash
     pip install gdown
     ```
   * 아래 명령어를 사용하여 구글 드라이브의 대용량 파일들을 `/workspace`로 다운로드합니다:
     ```bash
     # 1) images.zip 다운로드 (3.6GB)
     gdown --id "1IPudDhknKMOHhMu5ROZtrPl7Oy9YRTKl" -O /workspace/images.zip

     # 2) pretrained_fashion_roberta.pth 다운로드 (500MB)
     gdown --id "1BgaPhphEFHhdWOAant2R2VQtD8KLqv1J" -O /workspace/pretrained_fashion_roberta.pth
     ```

---

### 💻 방법 B. SFTP / SCP 클라이언트 사용 (FileZilla 또는 WinSCP)
로컬 PC에서 RunPod 서버로 다이렉트 전송하는 방식입니다. 이어받기 기능이 있어 안정적입니다.

1. **연결 정보 확인**:
   * RunPod 대시보드에서 본인 Pod의 **`Connect`** 버튼을 누릅니다.
   * **`SSH Connection`** 정보(예: `ssh -p 12345 root@xx.xx.xx.xx`)를 확인합니다.
     * **호스트(Host)**: `xx.xx.xx.xx` (IP 주소)
     * **포트(Port)**: `12345` (포트 번호)
     * **사용자(Username)**: `root`
     * **비밀번호/키**: RunPod 가입 시 등록한 SSH Key 또는 설정된 패스워드

2. **FileZilla 설정**:
   * FileZilla를 실행하고 **[파일] -> [사이트 관리자]**로 이동합니다.
   * **프로토콜**: `SFTP - SSH File Transfer Protocol` 선택
   * **호스트**: RunPod IP 주소 입력
   * **포트**: RunPod 포트 번호 입력
   * **로그온 유형**: 본인의 SSH Key가 있다면 `키 파일`을 선택하고 등록하거나, 패스워드가 있을 경우 `일반`을 선택하여 비밀번호를 입력합니다.
   * **연결**을 누른 후 우측의 `/workspace/` 디렉토리로 로컬 파일들을 드래그 앤 드롭하여 업로드합니다.

---

### 📁 방법 C. 소형 파일 업로드 (JupyterLab GUI 브라우저 직접 사용)
코드 스크립트나 용량이 작은 CSV는 브라우저 화면에서 바로 업로드할 수 있어 편리합니다.

1. **대상 파일**:
   * `multimodal_mobile_version_v9_runpod.py` (29KB)
   * `fashion_train_subset_2_with_images.csv` (51MB)
   * `pretrained_fashion_mobilenet.pth` (12MB)
2. **업로드 방법**:
   * JupyterLab 왼쪽 사이드바 파일 브라우저 상단의 **`Upload files` (구름 모양에 위로 가는 화살표 아이콘)**을 클릭합니다.
   * 로컬의 `c:\Users\육태정\Desktop\BigData` 폴더에서 해당 파일들을 선택하여 업로드합니다.

---

## 🛠️ 3단계. 압축 해제 및 환경 구축하기 (JupyterLab Terminal)

파일 업로드가 완료되었다면, JupyterLab 터미널에서 다음 명령어들을 차례대로 실행하여 학습 대기를 마칩니다.

### 1. 이미지 압축 해제 (`unzip`)
RunPod의 미니멀 컨테이너에는 `unzip` 패키지가 없을 수 있으므로 설치 후 조용히 압축 해제합니다.
```bash
# apt 패키지 업데이트 및 unzip 설치
apt-get update && apt-get install -y unzip

# /workspace/images 폴더 경로에 압축 풀기 (-q 옵션으로 로그 출력 없이 빠르게 진행)
unzip -q /workspace/images.zip -d /workspace/
```
> [!NOTE]
> 압축이 풀리면 `/workspace/images/` 경로 아래에 수많은 이미지 파일들이 들어오게 되며, 파이썬 코드의 `replace("\\", "/")` 처리 덕분에 `images/파일명.jpg` 형태로 정상 로드됩니다.

### 2. 파이썬 라이브러리 추가 설치
RunPod의 PyTorch 기본 환경에 자연어 처리를 위한 `transformers`와 학습 가속을 위한 `accelerate`를 추가로 설치해 줍니다.
```bash
pip install transformers accelerate scikit-learn
```

---

## 🎯 4단계. 본격적인 Model V9 학습 실행하기

모든 준비가 끝났습니다! GPU를 활용하여 12시간 안전 정지 룰이 반영된 V9 모델 학습을 시작합니다.

### 1단계 사전학습(Fashion CLIP) 가중치를 적용하여 학습 시작
```bash
python /workspace/multimodal_mobile_version_v9_runpod.py \
    --data-csv /workspace/fashion_train_subset_2_with_images.csv \
    --text-pretrained /workspace/pretrained_fashion_roberta.pth \
    --image-pretrained /workspace/pretrained_fashion_mobilenet.pth \
    --output-dir /workspace/outputs \
    --epochs 10 \
    --phase1-epochs 2 \
    --batch-size 8 \
    --accum-steps 4 \
    --num-workers 4 \
    --max-hours 9.0
```
* **`--max-hours 11.5`**: 12시간 제한 전에 모델이 자동으로 중간 상태를 완벽히 저장하고 안전 종료합니다.
* **`--batch-size 8` / `--accum-steps 4`**: GPU 메모리(VRAM) 크기에 맞추어 배치 사이즈를 유동적으로 조정할 수 있습니다. (VRAM이 부족해 Out Of Memory(OOM)가 발생할 경우 배치 사이즈를 `4`나 `2`로 낮추고 `--accum-steps`를 각각 `8` 또는 `16`으로 높이시면 동일한 효과를 냅니다.)

### 학습 중단 시 즉시 복구 및 재개하기
혹시나 12시간 시간 초과 등으로 학습이 안전 종료되었을 경우, 처음부터 다시 돌릴 필요 없이 아래 명령어로 즉각 마지막 체크포인트부터 이어갑니다:
```bash
python /workspace/multimodal_mobile_version_v9_runpod.py \
    --data-csv /workspace/fashion_train_subset_2_with_images.csv \
    --output-dir /workspace/outputs \
    --resume
```

---

> [!TIP]
> **작업 시작 전 꿀팁!**
> 터미널에서 직접 실행 시 브라우저가 예기치 않게 닫히거나 네트워크가 끊겨도 백그라운드에서 계속 학습이 이어지도록 `nohup` 또는 `tmux`/`screen`을 쓰시는 것이 좋습니다.
> * 예: `nohup python /workspace/multimodal_mobile_version_v9_runpod.py [인자들...] > train.log 2>&1 &`
> * 이렇게 실행한 뒤 `tail -f train.log`로 실시간 로그를 감상하시면 안전합니다.
