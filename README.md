# SemEval-2026 Task 2 — Predicting Variation in Emotional Responses

## 📌 Project Description

This repository is dedicated to participation and system development for **SemEval-2026 Task 2: Predicting Variation in Emotional Responses**.  
The goal of the task is to **predict the variation of emotional responses** that different people might have when reading the same text.

---

## 🧩 Subtasks

- **Subtask A**: Given a text, predict the distribution of emotional responses (Valence and Arousal) across annotators.  
- **Subtask B**: Work with a test set containing two subsets of annotators: “labeled” and “unlabeled”. The model must generalize to unseen annotators and generate predictions for the unlabeled group.  

📖 More details: [Task Description](https://semeval2026task2.github.io/SemEval-2026-Task2/tasks)

---

## 📂 Data

The dataset includes:
- **Training set**: texts annotated with Valence & Arousal scores from multiple annotators  
- **Validation / development set**: same format as training  
- **Evaluation / test set**: gold annotations for the labeled group + required predictions for the unlabeled group  

Format: JSON / CSV / TSV (depending on the release).  
The official site provides schemas and validation scripts.  

🔗 [Data Info](https://semeval2026task2.github.io/SemEval-2026-Task2/data)

---

## ⚙️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2

# Install dependencies (example for Python)
pip install -r requirements.txt
