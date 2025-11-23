# Training Scripts

이 폴더는 모델 훈련 스크립트를 포함합니다.

---

## 📁 폴더 구조

```
data_train/
├── README.md                    # 이 파일
├── subtask1/                    # Subtask 1 훈련 (팀원)
│   ├── README.md
│   └── train_subtask1.py
└── subtask2a/                   # Subtask 2a 훈련 (당신)
    ├── README.md
    └── train_ensemble_subtask2a.py    # ✅ 훈련 완료
```

---

## 🎯 사용 시점

### Subtask 1 (팀원)
- `subtask1/train_subtask1.py` - 모델 훈련 (진행 중)

### Subtask 2a (당신)
- `subtask2a/train_ensemble_subtask2a.py` - ✅ 훈련 완료
  - 3개 모델 (seed 42, 123, 777)
  - 결과: CCC 0.5053, 0.5330, 0.6554

---

**마지막 업데이트**: 2025-11-23
