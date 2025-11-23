# Subtask 2a - Test Prediction

**담당자**: [당신 이름]
**태스크**: State Change Forecasting (Valence & Arousal 예측)

---

## 📁 현재 상태

이 폴더는 **비어있습니다** - 예측 스크립트는 이미 다른 곳에 있음

---

## ⭐ 중요: 예측 스크립트 위치

**현재 위치**: `scripts/data_analysis/subtask2a/predict_test_subtask2a.py`

이미 완성된 예측 스크립트가 `data_analysis/subtask2a/`에 있습니다!

---

## 🎯 사용 방법 (12월 중순)

### 옵션 1: 기존 스크립트 사용 (추천)

```bash
# 1. 테스트 데이터 다운로드
# test_subtask2a.csv → data/test/ 폴더에 저장

# 2. 예측 실행
python scripts/data_analysis/subtask2a/predict_test_subtask2a.py

# 3. 제출 파일 확인
# pred_subtask2a.csv 생성됨
```

### 옵션 2: 이 폴더에 복사/이동

```bash
# 필요시 스크립트를 여기로 이동
cp scripts/data_analysis/subtask2a/predict_test_subtask2a.py scripts/test/subtask2a/

# 실행
python scripts/test/subtask2a/predict_test_subtask2a.py
```

---

## 📊 입출력

**입력**:
- `data/test/test_subtask2a.csv` - 테스트 데이터
- `models/subtask2a_*.pt` - 훈련된 모델 3개
- `results/subtask2a/ensemble_results.json` - 앙상블 가중치

**출력**:
- `pred_subtask2a.csv` - 제출 파일

**형식**:
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
user_001,-0.1234,0.5678
user_002,0.2345,-0.3456
...
```

---

## 📝 추천

**옵션 1 (현재)**: `scripts/data_analysis/subtask2a/` 사용
- 장점: 이미 완성되어 있음, 한 곳에 모든 분석 스크립트
- 단점: 분석과 예측이 같은 폴더

**옵션 2**: 이 폴더로 이동
- 장점: 테스트 예측만 별도 관리
- 단점: 폴더 구조 변경 필요

**결론**: **옵션 1 추천** - 이미 잘 작동하는 스크립트를 그대로 사용

---

**마지막 업데이트**: 2025-11-23
**상태**: 준비 완료 (비어있음, 기존 스크립트 사용 권장)
**사용 시점**: 2025년 12월 중순
