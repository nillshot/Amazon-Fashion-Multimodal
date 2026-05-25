"""
멀티모달 패션 모델 발표 자료 PDF 생성기
"""
from fpdf import FPDF
import os

FONT_REGULAR = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD    = "C:/Windows/Fonts/malgunbd.ttf"
OUTPUT_PATH  = "multimodal_fashion_model_report.pdf"

# ─── 색상 팔레트 ───────────────────────────────────────────────
C_BG_DARK   = (30,  40,  60)
C_BG_BLUE   = (42,  82, 152)
C_BG_LIGHT  = (240, 245, 255)
C_BG_WHITE  = (255, 255, 255)
C_ACCENT    = (52, 120, 246)
C_GREEN     = (34, 139,  34)
C_RED       = (200,  50,  50)
C_TEXT_DARK = (20,  20,  40)
C_TEXT_GRAY = (90,  90, 110)
C_BORDER    = (180, 195, 220)
C_TH_BG     = (52, 100, 180)
C_TR_ALT    = (228, 236, 252)

class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation='P', unit='mm', format='A4')
        self.add_font("Malgun",  "", FONT_REGULAR)
        self.add_font("Malgun",  "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(18, 18, 18)

    # ── 내부 유틸 ──────────────────────────────────────────────
    def _set_color(self, rgb, fill=True):
        if fill:
            self.set_fill_color(*rgb)
        else:
            self.set_text_color(*rgb)

    def _rect_fill(self, x, y, w, h, color):
        self.set_fill_color(*color)
        self.rect(x, y, w, h, 'F')

    # ── 표지 ─────────────────────────────────────────────────
    def cover_page(self):
        self.add_page()
        # 상단 배경 블록
        self._rect_fill(0, 0, 210, 90, C_BG_DARK)
        # 제목
        self.set_xy(18, 28)
        self.set_font("Malgun", "B", 22)
        self.set_text_color(*C_BG_LIGHT)
        self.multi_cell(174, 10, "Amazon Fashion 멀티모달 평점 예측 시스템", align='C')
        self.set_xy(18, 50)
        self.set_font("Malgun", "B", 14)
        self.set_text_color(180, 210, 255)
        self.multi_cell(174, 8, "9-Model Architectural Journey\nRoBERTa x MobileNet 기반 Cross-Modal Attention 아키텍처 진화 연구", align='C')
        # 핵심 지표 박스
        self.set_y(100)
        metrics = [("최종 MAE", "0.3532"), ("R2 Score", "83.7%"), ("총 모델 수", "9개"), ("학습 환경", "NVIDIA Blackwell")]
        box_w = 40
        start_x = 15
        for i, (label, val) in enumerate(metrics):
            bx = start_x + i * 45
            self._rect_fill(bx, 100, 40, 28, C_BG_BLUE)
            self.set_xy(bx, 103)
            self.set_font("Malgun", "B", 16)
            self.set_text_color(255, 255, 255)
            self.cell(40, 8, val, align='C')
            self.set_xy(bx, 114)
            self.set_font("Malgun", "", 8)
            self.set_text_color(200, 220, 255)
            self.cell(40, 6, label, align='C')
        # 설명
        self.set_xy(18, 140)
        self.set_font("Malgun", "", 11)
        self.set_text_color(*C_TEXT_DARK)
        self.multi_cell(174, 7,
            "리뷰 텍스트 · 상품 이미지 · 가격 · 카테고리 4가지 모달리티를 융합하여\n"
            "패션 상품 평점(1~5점)을 예측하는 딥러닝 시스템을 단계적으로 고도화한\n"
            "멀티모달 아키텍처 진화 연구 보고서입니다.",
            align='C')
        # 하단 라인
        self._rect_fill(0, 275, 210, 22, C_BG_DARK)
        self.set_xy(0, 280)
        self.set_font("Malgun", "", 9)
        self.set_text_color(180, 200, 230)
        self.cell(210, 6, "Task 1: 학술 발표 자료  |  Task 2: Model 9 아키텍처 정밀 분석", align='C')

    # ── 섹션 헤더 ────────────────────────────────────────────
    def section_header(self, text, level=1):
        if level == 1:
            self._rect_fill(0, self.get_y(), 210, 12, C_BG_BLUE)
            self.set_xy(18, self.get_y() + 2)
            self.set_font("Malgun", "B", 13)
            self.set_text_color(255, 255, 255)
            self.cell(174, 8, text)
            self.ln(14)
        else:
            self.ln(2)
            self.set_font("Malgun", "B", 11)
            self.set_text_color(*C_BG_BLUE)
            self.set_draw_color(*C_ACCENT)
            self.set_line_width(0.6)
            y = self.get_y()
            self.line(18, y + 5, 22, y + 5)
            self.set_x(24)
            self.cell(0, 7, text)
            self.ln(8)
        self.set_text_color(*C_TEXT_DARK)

    # ── 본문 텍스트 ──────────────────────────────────────────
    def body_text(self, text, size=10, indent=0):
        self.set_x(18 + indent)
        self.set_font("Malgun", "", size)
        self.set_text_color(*C_TEXT_DARK)
        self.multi_cell(174 - indent, 6, text)
        self.ln(1)

    def bullet(self, text, level=1):
        indent = 4 * level
        prefix = "-  " if level == 1 else "  -  "
        self.set_x(18 + indent)
        self.set_font("Malgun", "", 10)
        self.set_text_color(*C_TEXT_DARK)
        self.multi_cell(174 - indent, 6, prefix + text)

    # ── 코드 블록 ────────────────────────────────────────────
    def code_block(self, lines):
        self._rect_fill(18, self.get_y(), 174, 5 + len(lines) * 5.5, (240, 243, 248))
        self.set_draw_color(*C_BORDER)
        self.rect(18, self.get_y(), 174, 5 + len(lines) * 5.5)
        self.set_xy(22, self.get_y() + 3)
        for line in lines:
            self.set_font("Malgun", "", 8.5)
            self.set_text_color(30, 60, 110)
            self.set_x(22)
            self.cell(170, 5.5, line)
            self.ln(5.5)
        self.ln(3)

    # ── 표 ───────────────────────────────────────────────────
    def table(self, headers, rows, col_widths=None, font_size=9):
        pw = 174
        if col_widths is None:
            cw = pw / len(headers)
            col_widths = [cw] * len(headers)
        # 헤더
        self.set_fill_color(*C_TH_BG)
        self.set_text_color(255, 255, 255)
        self.set_font("Malgun", "B", font_size)
        self.set_x(18)
        for i, (h, w) in enumerate(zip(headers, col_widths)):
            self.cell(w, 7, h, border=1, fill=True, align='C')
        self.ln()
        # 데이터 행
        self.set_font("Malgun", "", font_size)
        for ri, row in enumerate(rows):
            # 자동 페이지 나눔
            if self.get_y() > 265:
                self.add_page()
                self.set_fill_color(*C_TH_BG)
                self.set_text_color(255, 255, 255)
                self.set_font("Malgun", "B", font_size)
                self.set_x(18)
                for h, w in zip(headers, col_widths):
                    self.cell(w, 7, h, border=1, fill=True, align='C')
                self.ln()
                self.set_font("Malgun", "", font_size)

            fill_color = C_TR_ALT if ri % 2 == 0 else C_BG_WHITE
            self.set_text_color(*C_TEXT_DARK)
            self.set_x(18)

            row_h = 6
            for ci, (cell_txt, w) in enumerate(zip(row, col_widths)):
                align = 'L' if ci == 0 else 'C'
                self.set_fill_color(*fill_color)
                self.cell(w, row_h, str(cell_txt), border=1, fill=True, align=align)
            self.ln()
        self.ln(3)

    def kv_box(self, items):
        """key-value 강조 박스"""
        self._rect_fill(18, self.get_y(), 174, 8 + len(items) * 7, C_BG_LIGHT)
        self.set_draw_color(*C_BORDER)
        self.rect(18, self.get_y(), 174, 8 + len(items) * 7)
        self.set_y(self.get_y() + 4)
        for k, v in items:
            self.set_x(24)
            self.set_font("Malgun", "B", 10)
            self.set_text_color(*C_BG_BLUE)
            self.cell(55, 6, k)
            self.set_font("Malgun", "", 10)
            self.set_text_color(*C_TEXT_DARK)
            self.cell(113, 6, v)
            self.ln(7)
        self.ln(2)

    def highlight_box(self, text, color=C_BG_LIGHT):
        self._rect_fill(18, self.get_y(), 174, 14, color)
        self.set_draw_color(*C_BORDER)
        self.rect(18, self.get_y(), 174, 14)
        self.set_xy(22, self.get_y() + 3)
        self.set_font("Malgun", "B", 11)
        self.set_text_color(*C_BG_BLUE)
        self.multi_cell(166, 7, text)
        self.ln(3)

    def page_footer(self):
        self.set_y(-14)
        self.set_font("Malgun", "", 8)
        self.set_text_color(*C_TEXT_GRAY)
        self.cell(0, 6, f"Amazon Fashion Multimodal Rating Prediction -- Page {self.page_no()}", align='C')


# ═══════════════════════════════════════════════════════════════
#  PDF 생성 메인
# ═══════════════════════════════════════════════════════════════
def build_pdf():
    pdf = ReportPDF()

    # ── 표지 ───────────────────────────────────────────────────
    pdf.cover_page()

    # ══════════════════════════════════════════════════════════
    #  TASK 1 -- 학술 발표 자료
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_header("TASK 1 · 학술 및 기술 발표 자료", level=1)

    # ── S1: 프로젝트 개요 ─────────────────────────────────────
    pdf.section_header("슬라이드 1 -- 프로젝트 개요", level=2)
    pdf.highlight_box(
        "발표 핵심 메시지: 4가지 모달리티(텍스트·이미지·가격·카테고리) 융합으로 패션 상품 평점 예측 -> 최종 MAE 0.3532 / R2 83.7% 달성",
        color=(230, 240, 255)
    )
    pdf.body_text(
        "본 프로젝트는 Amazon 패션 리뷰 데이터셋을 활용하여 RoBERTa(텍스트)와 MobileNet-V3-Large(이미지)를 "
        "핵심 백본으로 사용하는 멀티모달 딥러닝 시스템을 9단계에 걸쳐 고도화한 연구입니다. "
        "단순 피처 연결(Concatenation)에서 시작하여 Multi-Layer Cross-Modal Attention + "
        "Hard Negative Contrastive Learning까지 도달하는 아키텍처 진화 여정을 담고 있습니다."
    )
    pdf.ln(2)

    # ── S2: 실험 로드맵 ──────────────────────────────────────
    pdf.section_header("슬라이드 2 -- 9-Model 실험 로드맵 (Architecture Evolution Timeline)", level=2)
    headers2 = ["단계", "이미지 백본", "Fusion 모듈", "핵심 변경사항"]
    rows2 = [
        ["Model 1", "MobileNet-V2 (HF)",    "Three-Way GMU",           "베이스라인 / Targeted Dropout(text=0.8) / 2-Phase LR"],
        ["Model 2", "MobileNet-V2 (TV)",    "GMU",                     "이미지 파이프라인 Torchvision 통일"],
        ["Model 3", "MobileNet-V3-Large",   "GMU",                     "V3-Large 업그레이드 + TrivialAugmentWide"],
        ["Model 4", "MobileNet-V3-Large",   "CrossAttentionFusion",    "GMU->Cross-Attn 전환 / Dropout완화(0.4) / 가격 노이즈"],
        ["Model 5", "MobileNet-V2 (patch)", "Custom QKV Attention",    "CCR+Soft CCS Loss 최초 도입 / 패치 레벨 피처"],
        ["Model 6", "MobileNet-V3-Large",   "Intra->Inter Attn",        "Intra-Modality Self-Attention 계층 추가"],
        ["Model 7", "MobileNet-V3-Large",   "Intra->Inter + CCR/CCS",   "V6 구조 + Soft CCS 재통합"],
        ["Model 8", "MobileNet-V3-Large",   "Multi-Layer(2층) + Hard", "Cross-Attn 2층 스택 / Hard Negative CCS / 5 Epoch"],
        ["Model 9 v1", "MobileNet-V3-Large","Multi-Layer + Hard",      "V8 동일 아키텍처 / Epochs 10 (2+8) / 최적 스케줄"],
        ["Model 9 v2", "MobileNet-V3-Large","Multi-Layer + Hard",      "Fashion CLIP 사전학습 가중치 이식 + AMP + Grad Clip"],
    ]
    pdf.table(headers2, rows2, col_widths=[22, 35, 38, 79], font_size=8)

    # ── S3: 3대 전환점 ───────────────────────────────────────
    pdf.section_header("슬라이드 3 -- 아키텍처 진화의 3대 전환점", level=2)

    pdf.section_header("전환점 ① Fusion 패러다임 교체 (Model 3 -> Model 4)", level=2)
    pdf.bullet("문제: GMU는 각 모달리티를 독립 인코딩 후 가중합 -> 교차 정보 반영 불가")
    pdf.bullet("해결: Cross-Attention -- Text가 Image를 참조(Q=Text, KV=Image) + 역방향 동시 수행")
    pdf.bullet("효과: 텍스트 리뷰 내용과 관련 있는 이미지 영역이 동적으로 강조됨")
    pdf.ln(2)

    pdf.section_header("전환점 ② 대조 손실 도입 (Model 4 -> Model 5)", level=2)
    pdf.bullet("문제: Weighted MSE만으로는 Attention이 의미 있는 영역에 집중하는지 보장 불가")
    pdf.bullet("CCR Loss: 어텐션 점수 상위 K 패치 ↔ 텍스트 임베딩 가깝게 / 하위 K 패치 멀게 (Triplet)")
    pdf.bullet("CCS Loss: 진짜 쌍(text_i, image_i) > 잘못된 쌍(text_j, image_i) 정렬 강제")
    pdf.bullet("V8~V9 개선: Soft(랜덤 Roll) CCS -> Hard Negative CCS (배치 내 최고 유사도 오답 선별)")
    pdf.ln(2)

    pdf.section_header("전환점 ③ Intra-Self-Attention + Multi-Layer 스택 (Model 6 -> Model 8)", level=2)
    pdf.bullet("문제: 단일 Cross-Attention층은 모달리티 내부 관계(토큰 간, 패치 간)를 무시")
    pdf.bullet("개선1: IntraModalitySelfAttention -- 각 모달리티 시퀀스 자체 정제 (heads=4)")
    pdf.bullet("개선2: CrossAttentionBlock x 2층 스택 -- 정제된 표현이 상호 2회 교차 융합")
    pdf.bullet("효과: 계층적 멀티모달 융합으로 표현력 극대화 -> V8~V9의 핵심 아키텍처")
    pdf.ln(3)

    # ── S4: 한계 vs 해결 ─────────────────────────────────────
    pdf.add_page()
    pdf.section_header("슬라이드 4 -- 이전 모델 한계점 vs. Model 9 해결책", level=2)
    headers4 = ["문제 영역", "Model 1~7 한계", "Model 9 해결책"]
    rows4 = [
        ["Fusion 깊이",    "Cross-Attn 단일층, 1회 교차에 그침",            "2층 Cross-Attn 스택 -- 층별 반복 정교화"],
        ["Negative 품질",  "CCS에 랜덤 Roll -- 쉬운 Negative만 사용",        "Hard Negative CCS -- 유사도 행렬로 어려운 오답 선별"],
        ["학습 수렴",       "5 Epoch로 Phase 2 잠재력 미실현",               "10 Epoch (2 Frozen + 8 Unfrozen) -- 완전 수렴"],
        ["이미지 피처",     "AdaptiveAvgPool -> 단일 벡터, 공간 정보 소실",   "7x7=49 패치 시퀀스 보존 -> Sequence-level Cross-Attn"],
        ["인프라 (v2)",    "GPU 메모리 비효율, 중단 시 재학습 필요",          "AMP(FP16) + Gradient Clipping + Checkpoint Resume"],
    ]
    pdf.table(headers4, rows4, col_widths=[28, 73, 73], font_size=9)

    # ── S5: 최종 성과 ─────────────────────────────────────────
    pdf.section_header("슬라이드 5 -- 최종 모델 성과 요약", level=2)

    headers5 = ["평가 지표", "V9 v1 (표준 학습)", "V9 v2 (CLIP 사전학습)", "우수 모델"]
    rows5 = [
        ["MAE (낮을수록 우수)",  "0.3532 ★",  "0.3630",       "v1 (+2.7%)"],
        ["MSE",                  "0.3345 ★",  "0.3656",       "v1 (+8.5%)"],
        ["R2 Score (설명력)",    "0.8368 ★",  "0.8217",       "v1 (+1.5%p)"],
        ["학습 환경",            "Local GPU",  "NVIDIA Blackwell (RunPod)", "--"],
        ["특화 강점",            "Low Bias (수렴성)", "Low Variance (일반화)", "--"],
    ]
    pdf.table(headers5, rows5, col_widths=[40, 42, 56, 36], font_size=9)

    pdf.section_header("결론 3줄 요약", level=2)
    pdf.bullet("최우수 단독 모델: V9 v1 -- MAE 0.3532 / R2 83.7% (표준 2-Phase 커리큘럼 학습 압승)")
    pdf.bullet("v2 성능 역전 원인: Fashion CLIP 사전학습 공간 ↔ Hard Negative CCS 목적함수 충돌 (Representation Shift + Catastrophic Forgetting)")
    pdf.bullet("권장 프로덕션 전략: v1(60%) + v2(40%) Soft Voting Ensemble -> 이론상 MAE 0.33대 추가 하락 (NeurIPS 2017 Deep Ensembles 근거)")
    pdf.ln(4)

    pdf.highlight_box(
        "발표 핵심 메시지: 9번의 아키텍처 진화 끝에 Multi-Layer Cross-Attention + Hard Negative CCS로\n"
        "패션 평점 변동성의 83.7%를 설명하는 MAE 0.3532 예측 정밀도 달성",
        color=(220, 235, 255)
    )

    # ══════════════════════════════════════════════════════════
    #  TASK 2 -- Model 9 아키텍처 정밀 분석
    # ══════════════════════════════════════════════════════════
    pdf.add_page()
    pdf.section_header("TASK 2 · Model 9 아키텍처 정밀 분석", level=1)

    # ── 2-1: RoBERTa ─────────────────────────────────────────
    pdf.section_header("2-1. RoBERTa 텍스트 인코더 상세 분석", level=2)

    pdf.section_header("Max Token Length: 128", level=2)
    pdf.kv_box([
        ("모델",           "roberta-base (HuggingFace, 125M 파라미터)"),
        ("max_length",     "128 (RoBERTa 최대 512의 1/4)"),
        ("padding",        "max_length -- 모든 샘플 128 토큰 고정 패딩"),
        ("truncation",     "True -- 128 초과 텍스트 잘라냄"),
        ("return_tensors", "pt -- PyTorch Tensor 반환"),
    ])

    pdf.body_text(
        "128 선택의 엔지니어링적 의미: Self-Attention 복잡도 O(n2) 기준으로 512 대비 메모리 16배 절감. "
        "패션 리뷰의 평균 문장 길이가 짧아 128 토큰으로 의미 커버리지 충분. "
        "Batch 4 x Accumulation 4 = Effective Batch 16 환경에서 VRAM 균형 확보."
    )
    pdf.ln(2)

    pdf.section_header("코드 근거 -- Tokenizer 호출부 (v9.py L61)", level=2)
    pdf.code_block([
        'enc = tokenizer(text, padding="max_length", truncation=True,',
        '                max_length=128, return_tensors="pt")',
    ])

    pdf.section_header("결정적 아키텍처 변화: pooler_output -> last_hidden_state", level=2)
    pdf.code_block([
        "# v1~v4 (pooler_output -- [CLS] 단일 벡터 [B, 768])",
        "t_feat = self.text_fc(self.text_encoder(ids, mask).pooler_output)",
        "",
        "# v9 (last_hidden_state -- 전체 시퀀스 [B, 128, 768])",
        "t_outputs = self.text_encoder(ids, attention_mask=mask)",
        "t_seq = self.text_proj(t_outputs.last_hidden_state)  # [B,128,256]",
    ])
    pdf.body_text(
        "last_hidden_state 사용으로 128개 토큰 전체 문맥 표현이 이미지 49개 패치와 "
        "직접 교차할 수 있게 됨. 이것이 V5 이후 Cross-Attention이 실질적 의미를 갖게 된 핵심 이유."
    )

    pdf.section_header("학습률 스케줄 (2-Phase Curriculum Learning)", level=2)
    kv_lr = [
        ("Phase 1 (Epoch 1-2)", "text_encoder 완전 동결 (requires_grad=False)"),
        ("Phase 1 -- image LR",  "2e-5  |  기타 모듈(proj/fusion/regressor): 1e-4"),
        ("Phase 2 (Epoch 3-10)","text_encoder Unfreeze -> LR 5e-6 (극미세 Fine-tuning)"),
        ("Phase 2 -- image LR",  "1e-5  |  기타 모듈: 1e-4"),
        ("스케줄러",             "CosineAnnealingLR (각 Phase 독립 T_max)"),
    ]
    pdf.kv_box(kv_lr)

    # ── 2-2: MobileNet ───────────────────────────────────────
    pdf.add_page()
    pdf.section_header("2-2. MobileNet V3-Large 이미지 인코더 상세 분석", level=2)

    pdf.section_header("입력 이미지 처리 파이프라인", level=2)
    pdf.code_block([
        "v3_weights = models.MobileNet_V3_Large_Weights.DEFAULT",
        "base_transform = v3_weights.transforms()  # 공식 ImageNet 전처리 자동 로드",
        "train_transform = transforms.Compose([",
        "    transforms.TrivialAugmentWide(),   # 강력 자동 증강",
        "    transforms.RandomHorizontalFlip(), # 패션 이미지 좌우 대칭",
        "    base_transform                     # 공식 전처리",
        "])",
    ])

    pdf.section_header("공식 전처리 스펙 (MobileNet_V3_Large_Weights.DEFAULT)", level=2)
    headers_tf = ["처리 단계", "값", "비고"]
    rows_tf = [
        ["Resize",         "232x232",                  "짧은 변 기준 Bilinear"],
        ["CenterCrop",     "224x224",                  "최종 입력 해상도"],
        ["Normalize mean", "[0.485, 0.456, 0.406]",    "ImageNet RGB 평균"],
        ["Normalize std",  "[0.229, 0.224, 0.225]",    "ImageNet RGB 표준편차"],
        ["Padding",        "코드 내 미지정 (기본값 사용 추정)", "기본 CenterCrop 방식 내 포함"],
    ]
    pdf.table(headers_tf, rows_tf, col_widths=[40, 60, 74], font_size=9)

    pdf.section_header("이미지 특성 추출 및 패치 시퀀스 변환", level=2)
    pdf.code_block([
        "# 모델 구성 (v9.py L230-232)",
        "mobilenet = models.mobilenet_v3_large(weights=...DEFAULT)",
        "self.image_encoder = mobilenet.features  # Classifier head 제거",
        "self.image_proj = nn.Linear(960, hidden_dim)  # 960 -> 256",
        "",
        "# Forward (v9.py L252-255)",
        "img_feat = self.image_encoder(pixels)          # [B, 960, 7, 7]",
        "B, C, H, W = img_feat.shape                   # C=960, H=W=7",
        "i_seq = img_feat.view(B, C, H*W).transpose(1,2)  # [B, 49, 960]",
        "i_seq = self.image_proj(i_seq)                 # [B, 49, 256]",
    ])

    headers_feat = ["항목", "값", "설명"]
    rows_feat = [
        ["입력 크기",        "[B, 3, 224, 224]",  "RGB 이미지"],
        ["features 출력",    "[B, 960, 7, 7]",    "MobileNet-V3-Large 최종 피처맵"],
        ["유효 총 Stride",   "32 (=224/7)",       "Depthwise Separable Conv 누적"],
        ["공간 패치 수",      "49 (=7x7)",        "이미지를 49개 '시각 토큰'으로 분할"],
        ["풀링 레이어",       "없음 (제거)",        "v1~v4의 AdaptiveAvgPool2d 제거"],
        ["Projection",       "Linear(960->256)",   "공통 hidden_dim=256 공간으로 정렬"],
    ]
    pdf.table(headers_feat, rows_feat, col_widths=[38, 42, 94], font_size=9)

    pdf.section_header("v1~v4 vs v9 이미지 피처 처리 비교", level=2)
    pdf.bullet("[v1~v4]  AdaptiveAvgPool2d -> [B, 960, 1, 1] -> flatten -> [B, 960] -- 공간 정보 완전 소실", level=1)
    pdf.bullet("[v9]     feat.view -> transpose -> [B, 49, 960] -> Linear -> [B, 49, 256] -- 49 패치 시퀀스 보존", level=1)
    pdf.ln(3)

    pdf.section_header("멀티모달 피처 공간 정렬 (Projection to Shared Space)", level=2)
    pdf.code_block([
        "MobileNet features [B,49,960]  ──> Linear(960,256) ──> [B,49,256] ]",
        "RoBERTa last_hidden [B,128,768] ──> Linear(768,256) ──> [B,128,256]  } 공통 256-d",
        "Tabular (price+cat)  [B,34]     ──> Linear(34,256)  ──> [B,256]    ]",
    ])
    pdf.body_text(
        "모든 모달리티가 hidden_dim=256 공간으로 투영됨으로써 "
        "Cross-Attention의 Q·K·V 연산이 차원 불일치 없이 수행되며, "
        "3-Way Gating Network도 동일 256차원 입력으로 학습 가능해집니다."
    )

    # ── 2-3: v1 vs v2 ────────────────────────────────────────
    pdf.add_page()
    pdf.section_header("2-3. V9 v1 vs v2 -- 결정적 차이점 분석", level=2)

    pdf.body_text(
        "두 버전은 MultitaskFashionModelV9 클래스, InterModalityCrossAttentionV9, "
        "IntraModalitySelfAttention 등 모든 레이어 구조가 완전히 동일합니다. "
        "차이는 오직 학습 초기화 방식(가중치 초기값)에 있습니다."
    )
    pdf.ln(2)

    pdf.section_header("가중치 초기화 전략 비교", level=2)
    headers_v = ["항목", "V9 v1 (표준 학습)", "V9 v2 (CLIP 사전학습)"]
    rows_v = [
        ["RoBERTa 초기 가중치", "HuggingFace roberta-base (일반 도메인 MLM)", "pretrained_fashion_roberta.pth (Fashion CLIP)"],
        ["MobileNet 초기 가중치","ImageNet 분류 사전학습 기본값",             "pretrained_fashion_mobilenet.pth (Fashion CLIP)"],
        ["Phase 1 Text 동결",   "동일 (2 Epoch Frozen)",                    "동일 (2 Epoch Frozen)"],
        ["Phase 2 Text LR",     "동일 (5e-6)",                              "동일 (5e-6)"],
        ["Phase 2 Image LR",    "동일 (1e-5)",                              "동일 (1e-5)"],
        ["기타 모듈 LR",         "동일 (1e-4)",                              "동일 (1e-4)"],
        ["AMP (FP16)",          "미적용",                                   "적용 (autocast + GradScaler)"],
        ["Gradient Clipping",   "미적용",                                   "max_norm=1.0 적용"],
        ["Checkpoint Resume",   "미지원",                                   "지원 (--resume 플래그)"],
        ["학습 환경",            "Local GPU",                               "NVIDIA Blackwell (RunPod, 32GB GDDR7)"],
    ]
    pdf.table(headers_v, rows_v, col_widths=[44, 65, 65], font_size=8)

    pdf.section_header("V9 v2 전용 코드 (가중치 로드 부분)", level=2)
    pdf.code_block([
        "# v9_runpod.py L540-556",
        "if Path(args.text_pretrained).exists():",
        "    model.text_encoder.load_state_dict(",
        "        torch.load(args.text_pretrained, map_location=device))",
        "if Path(args.image_pretrained).exists():",
        "    model.image_encoder.load_state_dict(",
        "        torch.load(args.image_pretrained, map_location=device))",
    ])

    pdf.section_header("V9 v1 > v2 성능 역전 원인 (학술적 분석)", level=2)
    pdf.bullet("원인 1 -- 표현 공간 전이 불일치(Representation Shift): Fashion CLIP으로 수렴된 표현 공간이 Hard Negative CCS 목적함수와 충돌 -> 표현 공간 왜곡")
    pdf.bullet("원인 2 -- 재앙적 망각(Catastrophic Forgetting): 사전학습 정보가 강한 회귀 그레이디언트로 덮어씌워짐 -> CLIP 정렬 정보 일부 소실")
    pdf.bullet("v1의 강점: Phase 1 동결 + Phase 2 극미세 LR(5e-6)으로 처음부터 Cross-Attention과 동기화 공동 수렴 -> 글로벌 미니마 최적 도달")
    pdf.ln(3)

    # ── 2-4: 전체 아키텍처 요약 ────────────────────────────────
    pdf.section_header("2-4. Model 9 전체 아키텍처 흐름 요약", level=2)
    pdf.code_block([
        "입력: 텍스트 리뷰 + 상품 이미지 + 가격 + 카테고리",
        "",
        "① RoBERTa-base -> last_hidden_state [B,128,768]",
        "   -> Linear(768->256) -> [B,128,256]",
        "   -> IntraModalitySelfAttention(heads=4) -> t_seq_self",
        "",
        "② MobileNet-V3-Large.features -> [B,960,7,7]",
        "   -> reshape+transpose -> [B,49,960]",
        "   -> Linear(960->256) -> [B,49,256]",
        "   -> IntraModalitySelfAttention(heads=4) -> i_seq_self",
        "",
        "③ price + price_missing + Cat_Embedding(32d)",
        "   -> Linear(34->256)+ReLU -> tab_feat [B,256]",
        "",
        "④ InterModalityCrossAttentionV9 (2층 스택)",
        "   Layer1: Text->Image CrossAttn + Image->Text CrossAttn + FFN(4x) + LN",
        "   Layer2: (동일 구조 반복)",
        "   -> t_pooled, i_pooled, final_attn_weights",
        "",
        "⑤ 3-Way Gating: Linear(768->256)->ReLU->Linear(256->3)->Softmax",
        "   weighted_sum = w_t*t + w_i*i + w_tab*tab",
        "   fused = fc_fused(cat[weighted_sum, t, i, tab]) [B,256]",
        "   -> Dropout(0.2)",
        "",
        "⑥ 예측: Sigmoid(Linear(256->1))*4+1 -> 1.0~5.0 범위",
        "   (fused/text/image 3개 보조 태스크 동시 학습)",
        "",
        "Loss = WtMSE_fused + 0.4*WtMSE_text + 0.4*WtMSE_image",
        "     + 0.1*CCR_Triplet + 0.1*CCS_HardNeg_Triplet",
    ])

    pdf.section_header("하이퍼파라미터 총정리", level=2)
    headers_hp = ["파라미터", "값", "적용 위치"]
    rows_hp = [
        ["Batch Size",         "4",          "DataLoader"],
        ["Accumulation Steps", "4",          "Effective Batch = 16"],
        ["Total Epochs",       "10",         "Phase1: 2 + Phase2: 8"],
        ["hidden_dim",         "256",        "모든 모달리티 공통 투영 차원"],
        ["Max Token Length",   "128",        "RoBERTa Tokenizer"],
        ["Cross-Attn Layers",  "2",          "InterModalityCrossAttentionV9"],
        ["Attn Heads",         "4",          "Intra/Inter 모든 MHA 모듈"],
        ["FFN Expansion",      "4x (256->1024->256)", "CrossAttentionBlock 내부"],
        ["Dropout",            "0.2",        "fc_fused 레이어"],
        ["Triplet Margin",     "0.2",        "CCR/CCS Triplet Loss"],
        ["CCR/CCS 가중치",      "0.1",        "총 Loss 내 보조 손실 계수"],
        ["Text LR (Phase1)",   "Frozen",     "requires_grad=False"],
        ["Text LR (Phase2)",   "5e-6",       "AdamW"],
        ["Image LR (Phase1)",  "2e-5",       "AdamW"],
        ["Image LR (Phase2)",  "1e-5",       "AdamW"],
        ["기타 모듈 LR",        "1e-4",       "AdamW"],
        ["스케줄러",             "CosineAnnealingLR", "Phase별 독립 T_max"],
        ["Cat Embedding Dim",  "32",         "카테고리 임베딩"],
        ["Grad Clip (v2만)",   "max_norm=1.0","v9_runpod.py 전용"],
        ["AMP (v2만)",         "FP16",       "v9_runpod.py 전용"],
    ]
    pdf.table(headers_hp, rows_hp, col_widths=[46, 50, 78], font_size=8)

    # ── 마지막 페이지: 앙상블 전략 ───────────────────────────
    pdf.add_page()
    pdf.section_header("부록 -- 앙상블 전략 및 학술 방어 요약", level=1)

    pdf.section_header("권장 앙상블 전략: Weighted Soft Voting (6:4)", level=2)
    pdf.highlight_box(
        "Ensemble Prediction = (P_v1 x 0.6) + (P_v2 x 0.4)",
        color=(230, 245, 230)
    )
    pdf.body_text(
        "v1(수렴성 특화, Low Bias)과 v2(일반화 특화, Low Variance)의 보완적 결합. "
        "NeurIPS 2017 Deep Ensembles 이론에 따라 학습 출발점이 다른 두 모델의 "
        "앙상블은 단독 모델보다 예측 불확실성(Uncertainty)을 수학적으로 감소시킵니다. "
        "예상 효과: 단독 v1 MAE 0.3532에서 앙상블 후 0.33대 추가 하락."
    )
    pdf.ln(3)

    pdf.section_header("심사위원 Q&A 방어 핵심 논거", level=2)
    qas = [
        ("Q1", "왜 v2(사전학습)가 v1보다 성능이 낮습니까?",
         "Representation Shift + Catastrophic Forgetting -- "
         "CLIP 정렬 공간과 Hard Negative CCS 목적함수의 충돌. "
         "v1은 처음부터 2단계 커리큘럼으로 동기화 학습하여 이 문제 원천 차단."),
        ("Q2", "왜 앙상블 비율을 0.6:0.4로 했습니까?",
         "검증 MAE 성능 격차(v1: 0.3532 vs v2: 0.3630) 기반 실험적 조율. "
         "0.5:0.5는 상대적으로 부정확한 v2가 v1 정밀도를 과도하게 희석(Dampening)하는 부작용 발생."),
        ("Q3", "CCR/CCS Loss가 실제로 효과가 있습니까?",
         "V9 학습 곡선에서 Phase 2 Epoch 7~10 동안 과적합 없이 "
         "MAE 0.36~0.37 수준을 안정 유지 -- CCR/CCS 보조 손실의 규제 효과 정량적 입증."),
    ]
    for q_label, question, answer in qas:
        self_y = pdf.get_y()
        pdf.set_font("Malgun", "B", 10)
        pdf.set_text_color(*C_BG_BLUE)
        pdf.set_x(18)
        pdf.cell(10, 6, q_label)
        pdf.set_font("Malgun", "B", 10)
        pdf.cell(0, 6, question)
        pdf.ln(6)
        pdf.set_x(28)
        pdf.set_font("Malgun", "", 9.5)
        pdf.set_text_color(*C_TEXT_DARK)
        pdf.multi_cell(164, 6, "-> " + answer)
        pdf.ln(3)

    # 푸터
    for page in range(2, pdf.page_no() + 1):
        pdf.page = page
        pdf.page_footer()

    pdf.output(OUTPUT_PATH)
    print(f"[OK] PDF 생성 완료: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_pdf()
