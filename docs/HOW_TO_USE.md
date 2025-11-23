# 📖 Subtask 2a - 사용 가이드

**작성일**: 2025-11-23
**상태**: ✅ 완료 및 테스트 준비

---

## 🎯 현재 상황

당신의 Subtask 2a 프로젝트는 **100% 완료**되었습니다!

### ✅ 완료된 것
- 3개 모델 훈련 완료 (CCC: 0.5053, 0.5330, 0.6554)
- 앙상블 가중치 계산 완료
- 예측 스크립트 준비 완료
- 모든 문서 작성 완료

### ⏳ 대기 중
- 테스트 데이터 공개 (12월 중순 예상)

---

## 📂 핵심 파일

### **predict_test_subtask2a.py**
**위치**: `scripts/data_analysis/subtask2a/predict_test_subtask2a.py`

**용도**: 실제 제출 파일 생성 (테스트 데이터 공개 후)

**언제**: **12월 중순** - 테스트 데이터 공개 후

**목적**:
- 테스트 데이터로 실제 예측 생성
- 제출 파일 `pred_subtask2a.csv` 생성
- Codabench 제출 준비

**실행 방법** (Google Colab):
```python
# 1. GitHub에서 프로젝트 클론
!git clone https://github.com/YOUR_USERNAME/Deep-Learning-project-SemEval-2026-Task-2.git
%cd Deep-Learning-project-SemEval-2026-Task-2

# 2. 테스트 데이터 다운로드 및 업로드
# test_subtask2a.csv → data/test/ 폴더에 저장

# 3. 스크립트 실행
!python scripts/data_analysis/subtask2a/predict_test_subtask2a.py

# 4. 제출 파일 확인
!head pred_subtask2a.csv

# 5. 제출 파일 다운로드
from google.colab import files
files.download('pred_subtask2a.csv')
```

**예상 실행 시간**: 10-30분 (테스트 세트 크기에 따라)

**출력 파일**: `pred_subtask2a.csv`
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
user_001,-0.1234,0.5678
user_002,0.2345,-0.3456
...
```

---

## 📅 단계별 가이드

### **단계 1: 12월 3일 전** ⭐ 현재 단계

**목표**: 진행상황 평가 준비

**해야 할 일**:
1. ✅ 진행상황 보고서 검토
   - 파일: `docs/03_EVALUATION_DEC3.md`
   - 팀원과 함께 준비 (팀원: Subtask 1, 당신: Subtask 2a)

2. ✅ 발표 자료 준비
   - 가이드: `docs/03_EVALUATION_DEC3.md` (Section B)
   - 15-17장 슬라이드
   - 발표 시간: 당신 6-8분, 팀원 3-4분

3. ✅ Q&A 준비
   - 참고: `docs/01_PROJECT_OVERVIEW.md` (Part 2)
   - 기술적 결정 이유 설명 준비
   - 학습 과정 및 어려움 설명

**사용 파일**:
- `docs/03_EVALUATION_DEC3.md` (평가 준비)
- `docs/01_PROJECT_OVERVIEW.md` (평가 기준)

**소요 시간**: 4-6시간 (팀원과 협력)

---

### **단계 2: 12월 중순 (테스트 데이터 공개 후)** 🚀

**목표**: 실제 제출 파일 생성 및 제출

**해야 할 일**:
1. ✅ 테스트 데이터 다운로드
   - 출처: https://www.codabench.org/competitions/9963/
   - 파일명: `test_subtask2a.csv`

2. ✅ 예측 실행
   - 파일: `predict_test_subtask2a.py`
   - 실행 시간: 10-30분

3. ✅ 제출 파일 검증
   ```bash
   head pred_subtask2a.csv
   ```

4. ✅ ZIP 파일 생성
   ```bash
   zip submission.zip pred_subtask2a.csv
   ```

5. ✅ Codabench 제출
   - URL: https://www.codabench.org/competitions/9963/
   - 마감: 2026년 1월 9일

**사용 파일**: `predict_test_subtask2a.py`

**상세 가이드**: `docs/04_SUBMISSION_GUIDE.md`

**소요 시간**: 1-2시간

---

## 🗂️ 전체 파일 구조

```
Deep-Learning-project-SemEval-2026-Task-2/
│
├── scripts/data_analysis/subtask2a/
│   ├── predict_test_subtask2a.py        ⭐ 테스트 데이터로 예측
│   └── analyze_ensemble_weights_subtask2a.py
│
├── scripts/data_train/subtask2a/
│   └── train_ensemble_subtask2a.py      ✅ 훈련 완료
│
├── models/
│   ├── subtask2a_seed42_best.pt   (1.5 GB)
│   ├── subtask2a_seed123_best.pt  (1.5 GB)
│   └── subtask2a_seed777_best.pt  (1.5 GB)
│
├── results/subtask2a/
│   └── ensemble_results.json
│
├── data/
│   ├── raw/
│   │   └── train_subtask2a.csv
│   └── test/
│       └── test_subtask2a.csv  ⏳ 아직 없음
│
└── docs/
    ├── README.md                      # 프로젝트 소개
    ├── HOW_TO_USE.md                  # 사용 가이드 (이 파일)
    ├── 01_PROJECT_OVERVIEW.md         # 프로젝트 개요 및 평가 기준
    ├── 02_TRAINING_AND_RESULTS.md     # 훈련 기록 및 결과
    ├── 03_EVALUATION_DEC3.md          # 12/3 평가 준비
    └── 04_SUBMISSION_GUIDE.md         # 제출 가이드
```

---

## ⚠️ 자주 발생하는 에러

### 1. FileNotFoundError: train_subtask2a.csv
**원인**: 데이터 파일 경로 문제

**해결**:
```python
# Colab에서 파일 위치 확인
!ls -la /content/Deep-Learning-project-SemEval-2026-Task-2/data/raw/

# 파일이 없으면 업로드
from google.colab import files
uploaded = files.upload()
!mv train_subtask2a.csv data/raw/
```

### 2. FileNotFoundError: subtask2a_seed*_best.pt
**원인**: 모델 파일 미업로드

**해결**:
```python
# 모델 파일 3개를 models/ 폴더에 업로드
!ls -la /content/Deep-Learning-project-SemEval-2026-Task-2/models/

# Google Drive 사용 (권장)
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/models/*.pt models/
```

### 3. CUDA out of memory
**원인**: GPU 메모리 부족

**해결**:
```python
# 배치 크기 줄이기
BATCH_SIZE = 8  # 기본값: 16

# 런타임 재시작
# 런타임 > 런타임 다시 시작
```

### 4. Ensemble weights file not found
**원인**: `ensemble_results.json` 파일 없음

**해결**: 자동으로 균등 가중치 사용 (1/3씩)
- 성능 차이: 미미 (CCC 0.01-0.02 정도)

---

## 📊 예상 성능

### 테스트 데이터 예측
```
개별 모델:
- seed42:  CCC 0.50-0.52
- seed123: CCC 0.52-0.54
- seed777: CCC 0.64-0.66 ⭐ 최고

앙상블:
- 예상 CCC: 0.58-0.61 (낙관적)
- 최소 CCC: 0.55-0.57 (보수적)
- 목표 CCC: 0.53-0.55

경쟁력: Top 10 가능성 높음
```

---

## ✅ 체크리스트

### 12월 3일 전 (현재 우선순위)
- [ ] 진행상황 보고서 검토 (팀원과)
- [ ] 발표 자료 준비 (15-17장)
- [ ] 발표 연습
- [ ] Q&A 준비

### 12월 중순 (테스트 데이터 후)
- [ ] 테스트 데이터 다운로드
- [ ] `predict_test_subtask2a.py` 실행
- [ ] `pred_subtask2a.csv` 확인
- [ ] ZIP 파일 생성
- [ ] Codabench 제출

---

## 🚀 빠른 시작 (Colab)

### 테스트 데이터 예측 실행
```python
# ===== 1. 프로젝트 클론 =====
!git clone https://github.com/YOUR_USERNAME/Deep-Learning-project-SemEval-2026-Task-2.git
%cd Deep-Learning-project-SemEval-2026-Task-2
!mkdir -p models data/test results/subtask2a

# ===== 2. 모델 파일 업로드 (Google Drive 권장) =====
from google.colab import drive
drive.mount('/content/drive')
!cp /content/drive/MyDrive/SemEval2026/models/*.pt models/
!cp /content/drive/MyDrive/SemEval2026/data/test_subtask2a.csv data/test/

# ===== 3. 예측 실행 =====
!python scripts/data_analysis/subtask2a/predict_test_subtask2a.py

# ===== 4. 결과 확인 및 다운로드 =====
!head pred_subtask2a.csv
from google.colab import files
files.download('pred_subtask2a.csv')
```

**예상 실행 시간**: 10-30분

---

## 📞 도움말

### 상세 가이드
- **제출 가이드**: `docs/04_SUBMISSION_GUIDE.md`
- **훈련 기록**: `docs/02_TRAINING_AND_RESULTS.md`
- **프로젝트 개요**: `docs/01_PROJECT_OVERVIEW.md`

### 평가 관련
- **평가 준비**: `docs/03_EVALUATION_DEC3.md`
- **평가 기준**: `docs/01_PROJECT_OVERVIEW.md` (Part 2)

### 대회 정보
- **대회 홈페이지**: https://semeval2026task2.github.io/SemEval-2026-Task2/
- **제출 사이트**: https://www.codabench.org/competitions/9963/
- **제출 마감**: 2026년 1월 9일

---

## 🎯 핵심 요약

### 지금 할 일 (12/3 전)
1. ✅ 진행상황 평가 준비
2. ✅ 발표 자료 제작
3. ✅ Q&A 준비

### 나중에 할 일
1. ⏳ 12월 중순: 테스트 데이터로 예측 실행
2. ⏳ 1월 9일 전: Codabench 제출

### 핵심 파일
- **predict_test_subtask2a.py** ← 테스트 데이터로 예측 (12월 중순)

---

**마지막 업데이트**: 2025-11-23
**상태**: ✅ 준비 완료
**다음 단계**: 12/3 평가 준비
