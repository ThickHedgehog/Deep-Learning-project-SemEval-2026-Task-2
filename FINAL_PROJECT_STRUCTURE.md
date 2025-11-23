# 📁 최종 프로젝트 구조

**작성일**: 2025-11-23
**상태**: ✅ 완료 - 최종 구조 확정
**목표**: Subtask 1 & 2a 완벽한 팀 협업 구조

---

## 🎯 전체 폴더 구조

```
Deep-Learning-project-SemEval-2026-Task-2/          (~4.35 GB)
│
├── 📄 README.md                          # 프로젝트 소개
├── 📄 requirements.txt                   # 의존성 패키지
├── 📄 .gitignore                         # Git 무시 파일
│
├── 📄 FOLDER_CLEANUP_PLAN.md             # 폴더 정리 계획서
├── 📄 PROJECT_CLEANUP_SUMMARY.md         # 정리 요약
├── 📄 FINAL_PROJECT_STRUCTURE.md         # 이 파일 ⭐
│
├── 📚 docs/                              # 문서 (6개)
│   ├── README.md                         # 프로젝트 소개
│   ├── HOW_TO_USE.md                     # 사용 가이드 ⭐
│   ├── 01_PROJECT_OVERVIEW.md            # 개요 및 평가 기준
│   ├── 02_TRAINING_AND_RESULTS.md        # 훈련 기록 (Subtask 2a)
│   ├── 03_EVALUATION_DEC3.md             # 12/3 평가 준비 ⭐⭐⭐
│   └── 04_SUBMISSION_GUIDE.md            # 제출 가이드
│
├── 📊 data/                              # 데이터 (~3 MB)
│   ├── raw/                              # 원본 데이터
│   │   ├── train_subtask1.csv            # Subtask 1 (557 KB)
│   │   └── train_subtask2a.csv           # Subtask 2a (579 KB)
│   ├── processed/                        # 전처리된 데이터
│   │   ├── subtask1_processed.csv        # Subtask 1 (1.7 MB)
│   │   └── subtask2a_features.csv        # Subtask 2a (2.1 MB)
│   └── test/                             # 테스트 데이터 (12월 중순)
│       ├── (test_subtask1.csv)           # 아직 없음
│       └── (test_subtask2a.csv)          # 아직 없음
│
├── 🤖 models/                            # 훈련된 모델 (4.3 GB)
│   ├── subtask2a_seed42_best.pt          # Subtask 2a 모델 1 (1.5 GB)
│   ├── subtask2a_seed123_best.pt         # Subtask 2a 모델 2 (1.5 GB)
│   ├── subtask2a_seed777_best.pt         # Subtask 2a 모델 3 (1.5 GB)
│   └── (subtask1 모델들)                  # 팀원이 추가 예정
│
├── 📈 results/                           # 훈련 결과
│   ├── subtask1/                         # Subtask 1 결과 (팀원용)
│   └── subtask2a/                        # Subtask 2a 결과
│       └── ensemble_results.json         # 앙상블 가중치
│
└── 💻 scripts/                           # 실행 스크립트
    │
    ├── 📊 data_analysis/                 # 데이터 분석 및 예측
    │   ├── README.md                     # 폴더 설명
    │   ├── subtask1/                     # Subtask 1 (팀원)
    │   │   ├── README.md
    │   │   └── analyze_raw_data_subtask1.py
    │   └── subtask2a/                    # Subtask 2a (당신)
    │       ├── README.md
    │       ├── predict_test_subtask2a.py          # 테스트 예측 ⭐
    │       ├── predict_test_subtask2a.ipynb
    │       └── analyze_ensemble_weights_subtask2a.py
    │
    ├── 🔧 data_preparation/              # 데이터 전처리
    │   ├── README.md                     # 폴더 설명
    │   ├── subtask1/                     # Subtask 1 (팀원)
    │   │   ├── README.md
    │   │   └── simple_data_prep_subtask1.py
    │   └── subtask2a/                    # Subtask 2a (당신)
    │       └── README.md                 # 전처리는 훈련에 통합됨
    │
    ├── 🎓 data_train/                    # 모델 훈련
    │   ├── README.md                     # 폴더 설명
    │   ├── subtask1/                     # Subtask 1 (팀원)
    │   │   ├── README.md
    │   │   └── train_subtask1.py
    │   └── subtask2a/                    # Subtask 2a (당신)
    │       ├── README.md
    │       └── train_ensemble_subtask2a.py        # ✅ 훈련 완료
    │
    └── 🧪 test/                          # 테스트 예측 (12월 중순)
        ├── README.md                     # 폴더 설명
        ├── subtask1/                     # Subtask 1 (팀원)
        │   └── README.md                 # 준비 완료 (비어있음)
        └── subtask2a/                    # Subtask 2a (당신)
            └── README.md                 # 준비 완료 (기존 스크립트 사용 권장)
```

---

## 📊 폴더별 상세 설명

### 1. 📚 docs/ - 문서

| 파일 | 용도 | 크기 | 중요도 |
|------|------|------|--------|
| README.md | 프로젝트 소개 (GitHub용) | 6.5 KB | ⭐ |
| HOW_TO_USE.md | 지금 뭘 해야 하는지 | 9.8 KB | ⭐⭐ |
| 01_PROJECT_OVERVIEW.md | 대회 규정 + 평가 기준 | 41.6 KB | ⭐ |
| 02_TRAINING_AND_RESULTS.md | 훈련 기록 (Subtask 2a) | 41.5 KB | ⭐ |
| 03_EVALUATION_DEC3.md | 12/3 평가 준비 | 35.2 KB | ⭐⭐⭐ |
| 04_SUBMISSION_GUIDE.md | 제출 가이드 | 16.8 KB | ⭐ |

**총 크기**: ~150 KB

---

### 2. 📊 data/ - 데이터

```
data/
├── raw/                # 원본 데이터 (1.1 MB)
│   ├── train_subtask1.csv       # Subtask 1: 감정 분류
│   └── train_subtask2a.csv      # Subtask 2a: 상태 변화 예측
│
├── processed/          # 전처리 데이터 (3.8 MB)
│   ├── subtask1_processed.csv   # Subtask 1 전처리
│   └── subtask2a_features.csv   # Subtask 2a 전처리
│
└── test/               # 테스트 데이터 (12월 중순)
    ├── test_subtask1.csv        # 아직 없음
    └── test_subtask2a.csv       # 아직 없음
```

**총 크기**: ~4.9 MB (test 데이터 제외)

---

### 3. 🤖 models/ - 훈련된 모델

```
models/
├── subtask2a_seed42_best.pt     # 1.5 GB (CCC 0.5053)
├── subtask2a_seed123_best.pt    # 1.5 GB (CCC 0.5330)
├── subtask2a_seed777_best.pt    # 1.5 GB (CCC 0.6554) ⭐ 최고
└── (subtask1 모델들)             # 팀원이 추가 예정
```

**총 크기**: 4.3 GB (Subtask 2a만)

---

### 4. 📈 results/ - 훈련 결과

```
results/
├── subtask1/                    # Subtask 1 결과 (팀원용)
│   └── (빈 폴더)                 # 팀원이 추가 예정
│
└── subtask2a/                   # Subtask 2a 결과
    └── ensemble_results.json    # 앙상블 가중치
                                 # {seed42: 0.298, seed123: 0.315, seed777: 0.387}
```

**총 크기**: ~10 KB

---

### 5. 💻 scripts/ - 실행 스크립트

#### 5.1 data_analysis/ - 분석 및 예측

```
data_analysis/
├── README.md
├── subtask1/ (팀원)
│   ├── README.md
│   └── analyze_raw_data_subtask1.py      # 데이터 탐색
│
└── subtask2a/ (당신)
    ├── README.md
    ├── predict_test_subtask2a.py         # 테스트 예측 ⭐
    ├── predict_test_subtask2a.ipynb
    └── analyze_ensemble_weights_subtask2a.py
```

#### 5.2 data_preparation/ - 전처리

```
data_preparation/
├── README.md
├── subtask1/ (팀원)
│   ├── README.md
│   └── simple_data_prep_subtask1.py      # 전처리
│
└── subtask2a/ (당신)
    └── README.md                         # 전처리는 훈련에 통합
```

#### 5.3 data_train/ - 훈련

```
data_train/
├── README.md
├── subtask1/ (팀원)
│   ├── README.md
│   └── train_subtask1.py                 # 훈련 (진행 중)
│
└── subtask2a/ (당신)
    ├── README.md
    └── train_ensemble_subtask2a.py       # ✅ 훈련 완료
```

#### 5.4 test/ - 테스트 예측 (12월 중순)

```
test/
├── README.md
├── subtask1/ (팀원)
│   └── README.md                         # 준비 완료 (비어있음)
│
└── subtask2a/ (당신)
    └── README.md                         # 기존 스크립트 사용 권장
                                          # (data_analysis/subtask2a/에 이미 있음)
```

---

## 👥 팀 협업 가이드

### Subtask 1 (팀원) 작업 영역 ✅

```
팀원의 파일:
├── data/raw/train_subtask1.csv
├── data/processed/subtask1_processed.csv
├── data/test/test_subtask1.csv (12월 중순)
├── scripts/data_analysis/subtask1/
├── scripts/data_preparation/subtask1/
├── scripts/data_train/subtask1/
├── scripts/test/subtask1/ (선택)
├── models/ (팀원 모델 저장)
└── results/subtask1/
```

### Subtask 2a (당신) 작업 영역 ✅

```
당신의 파일:
├── data/raw/train_subtask2a.csv
├── data/processed/subtask2a_features.csv
├── data/test/test_subtask2a.csv (12월 중순)
├── scripts/data_analysis/subtask2a/
├── scripts/data_preparation/subtask2a/
├── scripts/data_train/subtask2a/
├── scripts/test/subtask2a/ (선택)
├── models/subtask2a_*.pt (3개, 4.3 GB)
├── results/subtask2a/
└── docs/ (주로 관리)
```

### 공유 영역 ⚠️

```
수정 시 협의 필요:
├── README.md (프로젝트 소개)
├── requirements.txt (의존성)
├── docs/01_PROJECT_OVERVIEW.md (대회 규정)
└── .gitignore
```

---

## 📋 폴더 사용 타임라인

### 현재 (11월 23일)
```
✅ 사용 중:
├── data/raw/ (훈련 데이터)
├── data/processed/ (전처리 데이터)
├── models/ (Subtask 2a 모델 3개)
├── results/subtask2a/ (앙상블 가중치)
├── scripts/data_train/subtask2a/ (훈련 완료)
└── docs/ (문서)

⏳ 준비 중:
├── scripts/data_train/subtask1/ (팀원 훈련 중)
└── models/ (Subtask 1 모델 추가 예정)
```

### 12월 3일
```
📊 평가 준비:
└── docs/03_EVALUATION_DEC3.md (발표 자료)
```

### 12월 중순 (테스트 데이터 공개)
```
🚀 활성화:
├── data/test/ (테스트 데이터 다운로드)
├── scripts/data_analysis/subtask2a/predict_test_subtask2a.py
├── scripts/data_analysis/subtask1/ (팀원 예측)
└── scripts/test/ (선택적 사용)

📤 제출:
├── pred_subtask1.csv
└── pred_subtask2a.csv
```

---

## 📊 프로젝트 통계

### 파일 개수
```
전체: ~35개 파일
├── 문서: 9개 (docs/ + 루트)
├── Python 스크립트: 6개
├── README 파일: 12개
├── 데이터: 4개
├── 모델: 3개
└── 기타: 1개 (requirements.txt)
```

### 폴더 개수
```
전체: 22개 폴더
├── 최상위: 5개 (docs, data, models, results, scripts)
├── data: 3개
├── results: 2개
├── scripts: 12개
```

### 크기
```
전체: ~4.35 GB
├── models/       4.30 GB (99.0%)
├── data/         4.90 MB (0.1%)
├── docs/         150 KB (0.003%)
├── scripts/      80 KB (0.002%)
└── results/      10 KB (0.0002%)
```

---

## 🎯 핵심 파일 위치

### 지금 사용
| 파일 | 위치 | 용도 |
|------|------|------|
| 평가 준비 | docs/03_EVALUATION_DEC3.md | 12/3 발표 |
| 사용 가이드 | docs/HOW_TO_USE.md | 다음 단계 |
| 프로젝트 구조 | FINAL_PROJECT_STRUCTURE.md | 이 파일 |

### 12월 중순 사용
| 파일 | 위치 | 용도 |
|------|------|------|
| 테스트 예측 (2a) | scripts/data_analysis/subtask2a/predict_test_subtask2a.py | 제출 파일 생성 |
| 테스트 예측 (1) | scripts/data_analysis/subtask1/ | 팀원 추가 |
| 제출 가이드 | docs/04_SUBMISSION_GUIDE.md | Codabench 제출 |

---

## ✅ 체크리스트

### 폴더 구조 완성도
- [x] docs/ - 문서 통합 완료
- [x] data/ - 데이터 폴더 정리
- [x] models/ - Subtask 2a 모델 저장
- [x] results/ - Subtask 1 & 2a 분리
- [x] scripts/data_analysis/ - Subtask 1 & 2a 분리
- [x] scripts/data_preparation/ - Subtask 1 & 2a 분리
- [x] scripts/data_train/ - Subtask 1 & 2a 분리
- [x] scripts/test/ - 테스트 예측 폴더 생성
- [x] 모든 폴더에 README.md 추가

### 준비 완료
- [x] 팀 협업 구조 완성
- [x] Git 커밋 완료
- [x] 문서 작성 완료
- [x] 12월 중순 대비 폴더 생성

---

## 🚨 주의사항

### ❌ 절대 삭제/수정 금지
```
✗ models/subtask2a_*.pt              # 훈련 모델 (4.3 GB)
✗ results/subtask2a/ensemble_results.json
✗ scripts/data_analysis/subtask2a/predict_test_subtask2a.py
✗ data/raw/train_*.csv
✗ docs/
```

### ⚠️ 협의 후 수정
```
△ README.md
△ requirements.txt
△ docs/01_PROJECT_OVERVIEW.md
△ .gitignore
```

### ✅ 자유롭게 사용
```
✓ scripts/data_analysis/subtask1/
✓ scripts/data_preparation/subtask1/
✓ scripts/data_train/subtask1/
✓ scripts/test/subtask1/
✓ models/ (Subtask 1 모델 추가)
✓ results/subtask1/
```

---

## 📞 참고 문서

- **사용 가이드**: [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)
- **정리 계획**: [FOLDER_CLEANUP_PLAN.md](FOLDER_CLEANUP_PLAN.md)
- **정리 요약**: [PROJECT_CLEANUP_SUMMARY.md](PROJECT_CLEANUP_SUMMARY.md)
- **평가 준비**: [docs/03_EVALUATION_DEC3.md](docs/03_EVALUATION_DEC3.md)

---

**마지막 업데이트**: 2025-11-23
**상태**: ✅ 최종 구조 확정 - 프로젝트 끝까지 사용 가능
**다음 단계**: 12/3 평가 준비!
