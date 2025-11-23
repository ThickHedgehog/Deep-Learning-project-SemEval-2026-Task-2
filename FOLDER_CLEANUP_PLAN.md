# 📁 프로젝트 폴더 정리 완료

**작성일**: 2025-11-23
**상태**: ✅ 정리 완료
**목표**: Subtask 1 & 2a 팀 협업용 구조

---

## 🎯 정리 목표

1. ✅ **팀 협업 지원** - Subtask 1 (팀원) + Subtask 2a (당신)
2. ✅ **불필요한 파일 제거** - Subtask 2b 및 사용하지 않는 폴더
3. ✅ **명확한 구조** - 각 태스크별 폴더 분리
4. ✅ **Git 통합 준비** - 나중에 팀원과 통합 가능한 구조

---

## ✅ 실행된 작업

### 1. 삭제된 파일/폴더
```bash
✓ data/raw/train_subtask2b.csv                           # Subtask 2b
✓ data/raw/train_subtask2b_detailed.csv                  # Subtask 2b
✓ data/raw/train_subtask2b_user_disposition_change.csv   # Subtask 2b
✓ baselines/                                             # 사용 안함
✓ configs/                                               # 사용 안함
✓ src/                                                   # 사용 안함
✓ tests/                                                 # 사용 안함
✓ data/train/ (빈 폴더)                                  # 빈 폴더
```

### 2. 생성된 폴더/파일
```bash
✓ scripts/data_preparation/subtask2a/                    # Subtask 2a 전처리
✓ scripts/data_preparation/subtask2a/README.md
✓ scripts/data_analysis/README.md
✓ scripts/data_preparation/README.md
✓ scripts/data_train/README.md
✓ results/subtask1/                                      # Subtask 1 결과
✓ data/test/                                             # 테스트 데이터
```

### 3. 유지된 파일
```bash
✓ data/raw/train_subtask1.csv                            # Subtask 1 (팀원)
✓ data/raw/train_subtask2a.csv                           # Subtask 2a (당신)
✓ data/processed/subtask1_processed.csv                  # Subtask 1
✓ data/processed/subtask2a_features.csv                  # Subtask 2a
✓ scripts/data_analysis/analyze_raw_data_subtask1.py     # Subtask 1
✓ scripts/data_preparation/simple_data_prep_subtask1.py  # Subtask 1
✓ scripts/data_train/train_subtask1.py                   # Subtask 1
✓ 모든 Subtask 2a 스크립트 및 모델
```

---

## 🗂️ 최종 폴더 구조

```
Deep-Learning-project-SemEval-2026-Task-2/
│
├── .git/                              # Git 저장소
├── .gitignore                         # Git 무시 파일
├── README.md                          # 프로젝트 소개
├── requirements.txt                   # 의존성
├── FOLDER_CLEANUP_PLAN.md             # 이 파일
│
├── docs/                              # 📚 문서 (주로 Subtask 2a)
│   ├── README.md                      # 프로젝트 소개
│   ├── HOW_TO_USE.md                  # 사용 가이드
│   ├── 01_PROJECT_OVERVIEW.md         # 프로젝트 개요 및 평가 기준
│   ├── 02_TRAINING_AND_RESULTS.md     # 훈련 기록 및 결과 (Subtask 2a)
│   ├── 03_EVALUATION_DEC3.md          # 12/3 평가 준비
│   └── 04_SUBMISSION_GUIDE.md         # 제출 가이드
│
├── data/                              # 📊 데이터
│   ├── raw/
│   │   ├── train_subtask1.csv         # Subtask 1 원본 데이터 (팀원)
│   │   └── train_subtask2a.csv        # Subtask 2a 원본 데이터 (당신)
│   ├── processed/
│   │   ├── subtask1_processed.csv     # Subtask 1 전처리 데이터
│   │   └── subtask2a_features.csv     # Subtask 2a 전처리 데이터
│   └── test/                          # 테스트 데이터 (12월 중순)
│       ├── (test_subtask1.csv)        # 아직 없음
│       └── (test_subtask2a.csv)       # 아직 없음
│
├── models/                            # 🤖 훈련된 모델
│   ├── subtask2a_seed42_best.pt       # Subtask 2a 모델 1 (1.5 GB)
│   ├── subtask2a_seed123_best.pt      # Subtask 2a 모델 2 (1.5 GB)
│   ├── subtask2a_seed777_best.pt      # Subtask 2a 모델 3 (1.5 GB)
│   └── (subtask1 모델들)               # 팀원이 추가 예정
│
├── results/                           # 📈 훈련 결과
│   ├── subtask1/                      # Subtask 1 결과 (팀원)
│   └── subtask2a/
│       └── ensemble_results.json      # Subtask 2a 앙상블 가중치
│
└── scripts/                           # 💻 실행 스크립트
    ├── data_analysis/
    │   ├── README.md                  # 분석 스크립트 설명
    │   ├── analyze_raw_data_subtask1.py       # Subtask 1 분석 (팀원)
    │   └── subtask2a/
    │       ├── README.md
    │       ├── predict_test_subtask2a.py      # Subtask 2a 예측 (당신)
    │       ├── predict_test_subtask2a.ipynb
    │       └── analyze_ensemble_weights_subtask2a.py
    │
    ├── data_preparation/
    │   ├── README.md                  # 전처리 스크립트 설명
    │   ├── simple_data_prep_subtask1.py       # Subtask 1 전처리 (팀원)
    │   └── subtask2a/
    │       └── README.md              # Subtask 2a는 훈련에 통합
    │
    └── data_train/
        ├── README.md                  # 훈련 스크립트 설명
        ├── train_subtask1.py          # Subtask 1 훈련 (팀원)
        └── subtask2a/
            ├── README.md
            └── train_ensemble_subtask2a.py    # Subtask 2a 훈련 (당신)
```

---

## 👥 팀 협업 구조

### Subtask 1 (팀원)
```
📂 팀원의 작업 영역:
├── data/raw/train_subtask1.csv
├── data/processed/subtask1_processed.csv
├── scripts/data_analysis/analyze_raw_data_subtask1.py
├── scripts/data_preparation/simple_data_prep_subtask1.py
├── scripts/data_train/train_subtask1.py
├── models/ (팀원이 훈련한 모델)
└── results/subtask1/
```

### Subtask 2a (당신)
```
📂 당신의 작업 영역:
├── data/raw/train_subtask2a.csv
├── data/processed/subtask2a_features.csv
├── scripts/data_analysis/subtask2a/
├── scripts/data_train/subtask2a/
├── models/subtask2a_*.pt (3개 모델)
├── results/subtask2a/
└── docs/ (대부분 Subtask 2a 문서)
```

### 공유 영역
```
📂 공유 파일:
├── README.md
├── requirements.txt
├── data/test/ (12월 중순 테스트 데이터)
└── docs/01_PROJECT_OVERVIEW.md (대회 규정 및 평가 기준)
```

---

## 🔄 Git 통합 가이드

### 팀원과 통합 시 권장 사항

1. **브랜치 전략**:
   ```bash
   main
   ├── subtask1  (팀원 브랜치)
   └── subtask2a (당신 브랜치)
   ```

2. **충돌 방지**:
   - 각자 자신의 태스크 폴더만 수정
   - 공유 파일(README.md, requirements.txt)은 사전 협의 후 수정
   - docs/ 폴더: 주로 당신이 관리, 팀원은 Subtask 1 섹션만 추가

3. **머지 전 체크리스트**:
   ```bash
   # 팀원 코드 확인
   - [ ] Subtask 1 스크립트가 제대로 작동하는가?
   - [ ] 모델 파일이 models/ 폴더에 있는가?
   - [ ] requirements.txt에 필요한 패키지가 추가되었는가?

   # 당신의 코드 확인
   - [ ] Subtask 2a 스크립트가 영향받지 않는가?
   - [ ] 문서가 업데이트되었는가?
   - [ ] 모델 파일 경로가 유지되는가?
   ```

4. **머지 명령어**:
   ```bash
   # 메인 브랜치로 이동
   git checkout main

   # 팀원 브랜치 머지
   git merge subtask1

   # 당신 브랜치 머지
   git merge subtask2a

   # 충돌 해결 후
   git commit -m "Merge Subtask 1 and Subtask 2a"
   git push origin main
   ```

---

## 📊 프로젝트 통계

### 파일 개수
```
이전: ~40개 파일
현재: ~30개 파일
감소: 10개 파일 (25% 감소)
```

### 폴더 크기
```
전체: ~4.35 GB
├── models/       4.3 GB (99%)
├── data/         ~3 MB (0.07%)
├── docs/         ~150 KB
└── scripts/      ~80 KB
```

### 삭제된 파일 크기
```
Subtask 2b 데이터: ~1.8 MB
기타 폴더: ~5 KB
총 절약: ~1.8 MB
```

---

## ✅ 체크리스트

### 정리 완료 항목
- [x] Subtask 2b 파일 삭제
- [x] 사용하지 않는 폴더 삭제 (baselines, configs, src, tests)
- [x] 팀 협업용 폴더 생성 (results/subtask1, data/test)
- [x] README 파일 추가 (scripts 하위 폴더)
- [x] 최종 폴더 구조 문서화

### 유지 관리
- [ ] 팀원과 Git 브랜치 전략 협의
- [ ] 테스트 데이터 공개 시 data/test/에 저장
- [ ] 팀원 모델 파일 추가 시 models/ 폴더 사용
- [ ] 최종 제출 전 통합 테스트

---

## 🎯 다음 단계

### 1. 12월 3일 전 (현재 우선순위)
- 진행상황 평가 준비
- 발표 자료 제작 (팀원과 협력)
- Q&A 준비

### 2. 팀원과 협업
- Git 브랜치 전략 논의
- 공유 파일 수정 규칙 정의
- 통합 테스트 계획

### 3. 12월 중순 (테스트 데이터 후)
- 테스트 데이터 다운로드 → data/test/
- 각자 예측 실행
- 제출 파일 생성 및 검증

---

## 🚨 주의사항

### ❌ 절대 삭제하면 안 되는 것
```
✗ models/subtask2a_*.pt              # 당신의 훈련 모델 (4.3 GB)
✗ results/subtask2a/ensemble_results.json  # 앙상블 가중치
✗ scripts/data_analysis/subtask2a/predict_test_subtask2a.py  # 예측 스크립트
✗ data/raw/train_subtask1.csv        # 팀원 데이터
✗ data/raw/train_subtask2a.csv       # 당신 데이터
✗ docs/                              # 모든 문서
✗ scripts/data_train/train_subtask1.py  # 팀원 훈련 스크립트
```

### ⚠️ 공유 파일 수정 시 주의
```
△ README.md                          # 양쪽 태스크 설명 포함
△ requirements.txt                   # 양쪽 의존성 포함
△ docs/01_PROJECT_OVERVIEW.md        # 대회 규정 (공통)
```

---

## 📞 참고 문서

- **사용 가이드**: [docs/HOW_TO_USE.md](docs/HOW_TO_USE.md)
- **프로젝트 개요**: [docs/01_PROJECT_OVERVIEW.md](docs/01_PROJECT_OVERVIEW.md)
- **훈련 기록**: [docs/02_TRAINING_AND_RESULTS.md](docs/02_TRAINING_AND_RESULTS.md)
- **평가 준비**: [docs/03_EVALUATION_DEC3.md](docs/03_EVALUATION_DEC3.md)
- **제출 가이드**: [docs/04_SUBMISSION_GUIDE.md](docs/04_SUBMISSION_GUIDE.md)

---

**마지막 업데이트**: 2025-11-23
**상태**: ✅ 정리 완료 - 팀 협업 준비 완료
**다음 단계**: 12/3 평가 준비 + 팀원과 Git 전략 협의
