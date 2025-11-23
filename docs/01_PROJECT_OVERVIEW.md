# 01. Project Overview: SemEval 2026 Task 2

**Document Created**: 2025-11-23
**Purpose**: Complete overview of SemEval 2026 Task 2, evaluation criteria, and project context
**Contents**: Competition requirements + Professor's evaluation criteria + Subtask 2a overview

---

## 📋 Document Organization

This document integrates three essential sources:
1. **SemEval 2026 Task 2 Requirements** - Official competition guidelines
2. **Professor's Evaluation Criteria** - Course grading and expectations
3. **Subtask 2a Overview** - Our specific task focus

---

# PART 1: SemEval 2026 Task 2 Requirements

**Official Task**: Predicting Variation in Emotional Valence and Arousal over Time from Ecological Essays

---

## 🎯 Task Overview

### Background

**Objective**: Model emotion as a lived, dynamic experience rather than annotated perception

**Key Innovation**:
- First-person self-reported affect (not third-party annotations)
- Longitudinal data (2021-2024) from real-world settings
- Ecological essays and feeling words from U.S. service-industry workers
- Affective circumplex model: Valence (0-4) and Arousal (0-2)

### Affective Circumplex Model

```
Arousal (Activation)
      ↑ High (2)
      |
      |    Excited     Happy
      |       ·         ·
      |
Low ←-|------------------|--→ High    Valence (Pleasantness)
(0)   |                  |   (4)
      |       ·         ·
      |    Tense      Calm
      |
      ↓ Low (0)
```

**Valence Scale**:
- 0 = Highly negative affect
- 2 = Neutral
- 4 = Highly positive affect

**Arousal Scale**:
- 0 = Low energy
- 1 = Medium energy
- 2 = High energy

---

## 📊 Subtasks Description

### Subtask 1: Longitudinal Affect Assessment

**Input**: Sequence of m texts (essays or feeling words) in chronological order
```
e₁, e₂, ..., eₘ
```

**Output**: Valence & Arousal scores for each text
```
(v₁, a₁), (v₂, a₂), ..., (vₘ, aₘ)
```

**Test Split Groups**:
1. **Unseen users** – Users not observed during training
2. **Seen users** – Users in training set (but at future timesteps)

**Prediction File**: `pred_subtask1.csv`

---

### Subtask 2A: State Change Forecasting ⭐ (Current Focus)

**Input**:
- First t texts with their V & A scores
- Historical sequence up to time t

**Output**: Predict change from last observed timestep to next
```
Δ_state_valence = v_{t+1} - v_t
Δ_state_arousal = a_{t+1} - a_t
```

**Example**:
```
Time t:   valence = 2.0, arousal = 1.0
Time t+1: valence = 3.0, arousal = 1.5

Predicted state change:
  Δ_valence = +1.0  (moving more positive)
  Δ_arousal = +0.5  (slightly more energized)
```

**Prediction File**: `pred_subtask2a.csv`

**Current Status**: ✅ **Models Trained** (CCC 0.5846-0.6046 expected)

---

### Subtask 2B: Dispositional Change Forecasting

**Input**:
- First t texts with their V & A scores
- Historical average from time 1 to t

**Output**: Predict change from observed average to future average
```
Δ_dispo_valence = avg(v_{t+1:n}) - avg(v_{1:t})
Δ_dispo_arousal = avg(a_{t+1:n}) - avg(a_{1:t})
```

**Example**:
```
Past average (t=1 to 5):   valence = 2.0, arousal = 1.0
Future average (t=6 to 10): valence = 2.8, arousal = 1.3

Predicted dispositional change:
  Δ_valence = +0.8  (trending more positive over time)
  Δ_arousal = +0.3  (trending more energized)
```

**Prediction File**: `pred_subtask2b.csv`

**Current Status**: ❌ **Not Started**

---

## 📦 Submission Requirements

### File Format

**Required Archive**: `submission.zip`

**Structure**:
```
submission.zip
├── pred_subtask1.csv   (optional)
├── pred_subtask2a.csv  (optional)
└── pred_subtask2b.csv  (optional)
```

**Note**: You can submit 1, 2, or all 3 subtasks in a single ZIP file.

---

### Subtask 1 CSV Format

**Filename**: `pred_subtask1.csv`

**Required Columns** (exact order, header row required):
```csv
text_id,pred_valence,pred_arousal
```

**Example**:
```csv
text_id,pred_valence,pred_arousal
251,1.5,1.2
252,2.3,0.8
253,3.1,1.5
```

**Column Descriptions**:
- `text_id`: Integer ID from test dataset
- `pred_valence`: Float [0.0 - 4.0]
- `pred_arousal`: Float [0.0 - 2.0]

---

### Subtask 2A CSV Format

**Filename**: `pred_subtask2a.csv`

**Required Columns** (exact order, header row required):
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
```

**Example**:
```csv
user_id,pred_state_change_valence,pred_state_change_arousal
1,0.5,-0.2
3,-1.0,0.3
5,0.0,0.5
```

**Column Descriptions**:
- `user_id`: Integer user ID from test dataset
- `pred_state_change_valence`: Float (typically -4.0 to +4.0)
- `pred_state_change_arousal`: Float (typically -2.0 to +2.0)

**Important Notes**:
- Predictions are per **user**, not per text
- Represents change from last observed state to next state
- Can be negative (decreasing affect) or positive (increasing affect)

---

### Subtask 2B CSV Format

**Filename**: `pred_subtask2b.csv`

**Required Columns** (exact order, header row required):
```csv
user_id,pred_dispo_change_valence,pred_dispo_change_arousal
```

**Example**:
```csv
user_id,pred_dispo_change_valence,pred_dispo_change_arousal
1,0.3,0.1
3,-0.5,0.2
5,0.8,-0.1
```

**Column Descriptions**:
- `user_id`: Integer user ID from test dataset
- `pred_dispo_change_valence`: Float (typically -2.0 to +2.0)
- `pred_dispo_change_arousal`: Float (typically -1.0 to +1.0)

**Important Notes**:
- Predictions are per **user**, not per text
- Represents change in average affect over time
- Usually smaller magnitude than state changes

---

### Submission Process

1. **Prepare CSV files** with exact filenames and column orders
2. **Create ZIP archive** named `submission.zip`
3. **Upload to Codabench**: https://www.codabench.org/competitions/9963/
4. **Wait for evaluation results**

**Submission Deadline**: January 9, 2026 (11:55 PM GMT+1)

---

## 📝 Paper Requirements

### System Description Paper

**Required**: All participating teams must submit a system description paper

**Title Format**:
```
Team Name at SemEval-2026 Task 2: Descriptive Title
```

**Examples**:
- "Team at SemEval-2026 Task 2: RoBERTa-BiLSTM Ensemble for Affect Forecasting"
- "Team at SemEval-2026 Task 2: Temporal Modeling of Emotional State Changes"

**Important Notes**:
- Use "at" not "@"
- "SemEval-2026" (hyphen, no spaces)
- Task number followed by colon with space after
- NOT anonymous (include author names)

---

### Paper Length

**For Review Submission**:
- **Single task** (any number of subtasks): Up to **5 pages**
- **Multiple tasks** (same approach): 5 pages + 2 pages per additional task
  - 2 tasks: 7 pages
  - 3 tasks: 9 pages
- **Multiple tasks** (different approach): Separate 5-page papers

**Note**: Subtask 2A and 2B count as the **same task** (Task 2)

**Camera-Ready Version**:
- Add **1 additional page** to address reviewer feedback
- Single task: 6 pages
- Multiple tasks: varies

**Do NOT Count Toward Limit**:
- Acknowledgments
- References
- Appendices

---

### Paper Structure (Recommended)

#### 1. Abstract
Brief summary of task, approach, and results

#### 2. Introduction (~3 paragraphs)
- Task description and importance (cite task overview paper)
- Main strategy used
- Key findings and rankings
- Code release URL (if applicable)

#### 3. Background
- Task setup details (input/output)
- Dataset information (language, genre, size)
- Related work

#### 4. System Overview (longest section)
- Model architecture
- Key algorithms and decisions
- Resources used beyond training data
- Use equations, pseudocode, examples
- Explain preprocessing, feature engineering

#### 5. Experimental Setup
- Data splits usage
- Hyperparameter tuning
- External tools/libraries (with versions)
- Evaluation measures

#### 6. Results
- **Main Results**: Official performance and ranking
- **Quantitative Analysis**: Ablation studies, design comparisons
- **Error Analysis**: Confusion matrices, error subtypes, examples

**IMPORTANT**: Focus on analysis, not just rankings!

#### 7. Conclusion
- Summary of system and results
- Future work ideas

#### 8. Acknowledgments
- Grants, reviewers, collaborators

#### 9. Appendix (optional)
- Low-level implementation details
- Additional figures/results
- Hyperparameter values

---

### Paper Awards

**Best Paper Award**: Recognizes system description paper that:
- Advances understanding of the problem
- Has strong analysis component
- Clear and reproducible methodology
- Need NOT be the highest-scoring system

**Judging Criteria**:
- Scientific rigor
- Analysis depth
- Reproducibility
- Writing quality

---

## 📅 Important Dates

| Event | Date | Status |
|-------|------|--------|
| **Training Data Release** | ✅ Released | Complete |
| **Competition Opens** | ✅ Open | Active |
| **Test Data Release** | TBA | Pending |
| **Submission Deadline** | January 9, 2026 (11:55 PM GMT+1) | 49 days left |
| **Paper Submission Deadline** | TBA (after evaluation) | Pending |
| **Acceptance Notification** | TBA | Pending |
| **Camera-Ready Deadline** | TBA | Pending |
| **SemEval Workshop** | 2026 (exact date TBA) | Pending |

**Current Server Time**: November 20, 2025

---

## 🌐 Competition Platform

### Codabench

**Competition Page**: https://www.codabench.org/competitions/9963/

**Features**:
- Online submission system
- Automatic evaluation
- Real-time leaderboard
- Multiple submission attempts allowed

**Sign-Up Deadline**: January 9, 2026

**Docker Image**: `codalab/codalab-legacy:py37`

---

## ⚖️ Ethical Considerations

### Data Collection

- IRB approved study ("Data Science and Alcohol Consumption Study")
- Participants consented to public research use
- Anonymized data (no identifiable information)
- Named entities removed (persons, places, organizations)
- Contact information removed (phone, address, URL)
- Manual review performed after automated anonymization

### Model Limitations

**Important Warnings**:
- Automated systems ≠ objective measures of emotional state
- Language is subjective and culturally shaped
- Similar emotions expressed differently across individuals
- Predictions should be interpreted with caution
- Should NOT replace human judgment in sensitive applications

**Responsible Use**:
- Mental health: Complement, not replace, professional care
- Avoid oversimplification of complex emotions
- Consider individual and cultural context
- Be transparent about limitations

**Reference**: ACL 2022 Ethics Paper - https://aclanthology.org/2022.acl-long.573/

---

## 📚 Additional Resources

### Official Links

- **Task Website**: https://semeval2026task2.github.io/SemEval-2026-Task2/
- **Codabench**: https://www.codabench.org/competitions/9963/
- **GitHub**: (Check task website for official repository)

### SemEval Resources

- **SemEval Homepage**: https://semeval.github.io/
- **Paper Guidelines**: https://semeval.github.io/paper-requirements.html
- **System Paper Template**: https://semeval.github.io/system-paper-template.html
- **Past Workshops**: https://aclanthology.org/ (search "SemEval")

### Example Papers (Best Paper Winners)

- **SemEval-2022**: https://semeval.github.io/SemEval2022/awards
- **SemEval-2021**: https://semeval.github.io/SemEval2021/awards
- **SemEval-2020**: https://semeval.github.io/semeval2020-awards

---

## 📞 Contact

### Task Organizers

**Email**: nisoni@cs.stonybrook.edu

**Questions About**:
- Task clarifications
- Data issues
- Submission problems
- Evaluation questions

### Platform Issues

**Codabench Support**:
- GitHub: https://github.com/codalab/codabench
- Wiki: Check Codabench Wiki for platform documentation

---

# PART 2: Professor's Evaluation Criteria

**Course**: Deep Learning / NLP
**Project Type**: SemEval 2026 Task 2 - Team Project
**Participation Level**: General Participation (Not Competition-focused)

---

## 🎯 Evaluation Philosophy

### Core Principle

**"Individual Progress Over Absolute Results"**

> "I will evaluate individually how you contribute to the project.
> It is not the overall outcome. Rather your progress over time.
> What you learn from today until January."
>
> — Professor's Lecture

### What This Means

```
✅ EVALUATED:
- Your starting knowledge level
- What you learned during the project
- How you applied course concepts
- Your problem-solving process
- Individual contributions
- Growth over 3 months (Nov - Jan)

❌ NOT EVALUATED:
- Absolute competition ranking
- Team's overall performance only
- Comparison with top teams
- Whether you win awards
```

### Honesty Policy

**Professor's Emphasis**:
> "Be honest with me. Tell me the truth. Don't tell lies that you
> don't have any experience but you have experience and then you
> can't implement really great models."

**Implications**:
1. ✅ Declare your current skill level honestly
2. ✅ Show realistic progress from your baseline
3. ✅ Document your learning journey
4. ❌ Don't pretend to know more than you do
5. ❌ Don't hide your starting point

**Result**:
- Evaluation is **relative to YOUR starting point**
- Beginner showing great progress = High grade
- Expert doing minimum work = Low grade

---

## 📊 Evaluation Criteria

### 1. Individual Contribution (40%)

**What is Evaluated**:
```
✅ Code you wrote personally
✅ Experiments you designed
✅ Analysis you performed
✅ Documentation you created
✅ Problems you solved
✅ Team coordination efforts
```

**How to Document**:
```
Project Report Should Include:
1. "My Contributions" section
2. Specific code files/functions you wrote
3. Experiments you ran with results
4. Problems faced and solutions found
5. What you learned from each step

Example:
"I implemented the BiLSTM-Attention architecture
(scripts/data_train/subtask2a/train_ensemble_subtask2a.py,
lines 150-230). I designed the dual-head loss function
and conducted 15 experiments to optimize the CCC weights,
improving performance from 0.45 to 0.50 CCC."
```

**Evidence of Contribution**:
- Git commits (if using version control)
- Code comments with your name
- Experiment logs
- Meeting notes
- Email discussions

---

### 2. Learning & Progress (30%)

**What is Evaluated**:
```
Starting Point → End Point Growth

Examples:
- Never used PyTorch → Built full training pipeline
- Didn't know RNNs → Implemented BiLSTM
- No idea about NLP → Understood transformers
- Basic Python → Advanced deep learning code
```

**How to Demonstrate**:
```
In Your Report:

Section: "Learning Journey"

1. Initial Knowledge (Nov):
   "I had basic Python knowledge but never worked with
   deep learning frameworks. I didn't know what attention
   mechanisms were."

2. Learning Process:
   "Week 1-2: Studied PyTorch tutorials, implemented basic NN
    Week 3-4: Read attention papers, experimented with RoBERTa
    Week 5-6: Built ensemble system, optimized hyperparameters"

3. Final Capabilities:
   "By January, I can independently design and train
   transformer-based models, implement custom loss functions,
   and perform systematic ablation studies."

4. Challenges Overcome:
   "Initially struggled with GPU memory errors. Learned about
   gradient accumulation and batch size optimization."
```

**Documentation Tips**:
- Keep a learning journal (weekly notes)
- Screenshot error messages and solutions
- Save failed experiments (they show learning!)
- Document "aha moments"

---

### 3. Technical Implementation (20%)

**What is Evaluated**:
```
✅ Code quality and organization
✅ Model architecture design
✅ Experimental methodology
✅ Evaluation approach
✅ Reproducibility
```

**Quality Indicators**:

**Good Code** ✅:
```python
# Clear structure, documented, modular

class EmotionModel(nn.Module):
    """
    RoBERTa-based emotion prediction model.

    Architecture:
    - RoBERTa encoder (125M params)
    - BiLSTM (256 hidden, 2 layers)
    - Multi-head attention (8 heads)
    - Dual-head output (valence, arousal)

    Author: [Your Name]
    Date: 2025-11-15
    """
    def __init__(self, config):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.lstm = nn.LSTM(768, 256, num_layers=2, bidirectional=True)
        # ...
```

**Poor Code** ❌:
```python
# No comments, unclear naming, monolithic

def model(x):
    a = bert(x)
    b = lstm(a)
    c = att(b)
    return fc(c)
```

**Experimental Rigor**:
```
✅ Multiple random seeds (reproducibility)
✅ Train/val/test split properly used
✅ Hyperparameter search documented
✅ Ablation studies conducted
✅ Results tables with std dev
✅ Error analysis performed

❌ Single run, no validation
❌ Random hyperparameters, no justification
❌ Only reporting best result
❌ No analysis of failures
```

---

### 4. Analysis & Understanding (10%)

**What is Evaluated**:
```
✅ Understanding WHY things work/don't work
✅ Comparing different approaches
✅ Error analysis depth
✅ Insights from experiments
✅ Critical thinking
```

**Example Analysis** (Good):
```
Analysis Section:

"We observed that increasing Arousal CCC weight from 65% to 75%
decreased performance (CCC 0.50 → 0.28). This suggests that
over-emphasizing the harder task (arousal prediction) causes the
model to underfit on both tasks. The optimal weight of 70%
balances the two objectives.

Error Analysis:
- Model struggles with neutral emotions (valence 2.0): 45% error rate
- High-arousal predictions are underestimated: mean error -0.3
- User embeddings are critical: removing them drops CCC by 0.22

Hypothesis for future work:
User embeddings capture individual expression patterns. Consider
adding user demographics or personality features."
```

**Example Analysis** (Poor):
```
"We trained the model and got 0.50 CCC. It's good."
```

---

## 👥 Team Project Guidelines

### Team Composition

**Your Team**:
```
Team Size: 2 members
- Member 1 (Teammate): Subtask 1
- Member 2 (You): Subtask 2a

Task: SemEval 2026 Task 2
Participation: General (not competition-focused)
```

**Allowed Team Sizes**: 1-4 members

---

### Task Distribution Rules

**Professor's Guidance**:
```
✅ Pick ONE main task (e.g., Task 2)
✅ Can work on multiple subtasks within that task
✅ For 4-member team: Can split subtasks (2 on A, 2 on B)
✅ Can collaborate or work independently on subtasks
```

**Your Team's Distribution**:
```
Task 2: Predicting Variation in Emotional Responses
├── Subtask 1: Longitudinal Affect Assessment → Teammate
└── Subtask 2a: State Change Forecasting → You

Status:
- Subtask 1: In progress (teammate)
- Subtask 2a: ✅ Complete (CCC 0.5846-0.6046 expected)
```

---

### Collaboration Best Practices

#### 1. Regular Communication

**Minimum Requirements**:
```
- Weekly team meeting (30-60 min)
- Shared progress document
- Problem discussion channel (email/chat)
```

**Meeting Agenda Template**:
```
Week X Meeting - [Date]

1. Progress Updates (each member 5-10 min)
   - What you completed this week
   - Current results
   - Challenges faced

2. Technical Discussion (15-20 min)
   - Interesting findings
   - Code/approach to share
   - Questions for team/professor

3. Next Steps (10 min)
   - Individual goals for next week
   - Coordination needs
   - Resources to share

4. Administrative
   - Schedule next meeting
   - Action items
```

#### 2. Code Sharing

**Recommended Structure**:
```
Repository Layout:
├── shared/                    # Common utilities
│   ├── data_loader.py        # Shared by both subtasks
│   ├── evaluation.py         # Common metrics
│   └── preprocessing.py      # Data cleaning
│
├── subtask1/                 # Teammate's code
│   ├── model.py
│   ├── train.py
│   └── README.md
│
└── subtask2a/                # Your code
    ├── model.py
    ├── train.py
    └── README.md
```

**Code Review**:
- Review each other's code weekly
- Provide constructive feedback
- Learn from each other's approaches

#### 3. Documentation

**Shared Documentation**:
```
docs/
├── TEAM_PROGRESS.md          # Weekly updates
├── MEETING_NOTES.md          # Meeting summaries
├── TECHNICAL_DECISIONS.md   # Why we chose X over Y
└── INDIVIDUAL_CONTRIBUTIONS.md  # Who did what
```

**Example INDIVIDUAL_CONTRIBUTIONS.md**:
```markdown
# Individual Contributions

## [Your Name] - Subtask 2a

### Code Contributions
- Implemented RoBERTa-BiLSTM-Attention model
  - Files: scripts/data_train/subtask2a/train_ensemble_subtask2a.py
  - Lines: 1-500
  - Commit: abc123

- Designed dual-head loss function
  - Files: scripts/data_train/subtask2a/train_ensemble_subtask2a.py
  - Lines: 150-180
  - Innovation: Separate CCC/MSE weights for V and A

- Built ensemble system
  - Files: scripts/data_analysis/subtask2a/analyze_ensemble_weights_subtask2a.py
  - Strategy: Performance-based weighting

### Experiments Conducted
- Hyperparameter tuning: 15 experiments
- Ablation studies: User embeddings, attention, LSTM layers
- Ensemble: 3 models with different seeds

### Documentation
- ENSEMBLE_GUIDE.md (complete training guide)
- Technical architecture documentation
- Experiment logs and analysis

### Results
- Best single model: CCC 0.6554 (seed 777)
- Ensemble: CCC 0.5846-0.6046 (expected)

## [Teammate Name] - Subtask 1

[Teammate fills in their contributions]
```

---

## 📝 Paper Writing for Your Team

**Recommendation**: Write **ONE integrated paper** covering both subtasks

### Integrated Paper Structure (Recommended) ✅

**Title Format**:
```
"[Team Name] at SemEval-2026 Task 2: [Descriptive Title]"

Example:
"DataScience Team at SemEval-2026 Task 2: Temporal Modeling
of Emotional States with Ensemble Methods"
```

**Structure**:
```
1. Abstract (150-200 words)
   - Brief overview of both subtasks
   - Main approaches for each
   - Key results

2. Introduction (0.5-0.75 pages)
   - Task 2 importance and challenges
   - Subtask 1 vs 2a differences
   - Team approach overview
   - Paper organization

3. Background & Related Work (0.5 pages)
   - Emotion prediction literature
   - Longitudinal affect modeling
   - Related SemEval tasks

4. Task Description (0.5 pages)
   - Subtask 1: Longitudinal Affect Assessment
   - Subtask 2a: State Change Forecasting
   - Data characteristics
   - Evaluation metrics (CCC)

5. System Description - Subtask 1 (1-1.5 pages)
   [Teammate writes this section]
   - Model architecture
   - Training procedure
   - Hyperparameters
   - Implementation details

6. System Description - Subtask 2a (1-1.5 pages)
   [You write this section]
   - RoBERTa-BiLSTM-Attention architecture
   - Dual-head loss function
   - Ensemble strategy (3 models, performance weighting)
   - Feature engineering (39 features)
   - Training details

7. Experimental Setup (0.5 pages)
   - Common setup for both subtasks
   - Hardware (Google Colab T4 GPU)
   - Software versions
   - Data splits

8. Results (1 page)
   - Subtask 1 results [teammate]
   - Subtask 2a results [you]
     - Individual models: CCC 0.5053, 0.5330, 0.6554
     - Ensemble: CCC 0.5846-0.6046
   - Comparison table

9. Analysis (1 page)
   - Ablation studies (both subtasks)
   - Error analysis (both subtasks)
   - Comparative insights:
     * What worked well for both?
     * What was subtask-specific?
   - Failure cases

10. Conclusion (0.5 pages)
    - Summary of approaches
    - Key findings
    - Future work

Total: ~5 pages (within limit)

Acknowledgments:
"We thank Professor [Name] for guidance and PhD students
[Names] for technical support."

References: (not counted in limit)

Appendix (optional, not counted in limit):
- Hyperparameter values
- Additional ablation results
- Error examples
```

**Contribution Statement** (Include in Acknowledgments or as footnote):
```
"[Teammate Name] developed the Subtask 1 system and conducted
all related experiments. [Your Name] designed and implemented
the Subtask 2a ensemble approach and performed the analysis.
Both authors contributed equally to the paper writing, data
preprocessing, and overall project coordination."
```

---

### Paper vs Project Report

**For Professor** (Required):
```
Format: Academic project report
Length: No strict limit (10-15 pages typical)
Content:
- Detailed methodology
- All experiments (including failures)
- Learning process
- Individual contributions clearly marked
- Code appendix
- Extensive analysis

Purpose: Grade evaluation
```

**For SemEval** (Optional):
```
Format: ACL conference paper style
Length: 5 pages + references
Content:
- Polished results
- Main approach only
- Best experiments highlighted
- Concise writing

Purpose: Publication, community contribution
```

**Strategy**:
1. Write detailed project report first (for professor)
2. Condense to SemEval paper format if desired
3. Submit to SemEval as learning experience

---

## 📅 Timeline & Milestones

### Overall Timeline (Nov 2025 - Jan 2026)

```
Month 1: November 2025
├── Week 1 (Nov 4-10)
│   ├── Form team ✅
│   ├── Choose task ✅
│   ├── Download training data ✅
│   └── Initial data exploration ✅
│
├── Week 2 (Nov 11-17)
│   ├── Literature review
│   ├── Baseline model implementation
│   └── Initial experiments ✅
│
├── Week 3 (Nov 18-24)
│   ├── Model development ✅
│   ├── Feature engineering ✅
│   └── First real results ✅
│
└── Week 4 (Nov 25-30)
    ├── Progress meeting with professor
    ├── Model optimization
    └── Documentation

Month 2: December 2025
├── Week 5-6 (Dec 1-15)
│   ├── Advanced model development
│   ├── Ensemble methods
│   ├── Ablation studies
│   └── Error analysis
│
└── Week 7-8 (Dec 16-31)
    ├── Final model training
    ├── Test data preparation (if released)
    └── Draft paper/report

Month 3: January 2026
├── Week 9-10 (Jan 1-9)
│   ├── Test predictions
│   ├── Final submission (Jan 9 deadline)
│   └── Complete analysis
│
└── Week 11+ (Jan 10+)
    ├── Paper writing
    ├── Final report for professor
    └── Presentation preparation
```

---

### Your Team's Current Status (Nov 23, 2025)

```
✅ COMPLETED:
- Team formed (2 members)
- Task selected (Task 2)
- Subtasks distributed (1 and 2a)
- Training data downloaded and explored
- Subtask 2a: Full pipeline implemented
- Subtask 2a: 3 models trained
- Subtask 2a: Ensemble completed
- Expected results: CCC 0.5846-0.6046

🔄 IN PROGRESS:
- Subtask 1: Teammate working
- Documentation: Ongoing

⏳ UPCOMING:
- Late November: Progress meeting with professor
- Mid-December: Test data release (expected)
- January 9: Competition submission
- Mid-January: Final report to professor
```

---

### Key Milestones

#### Milestone 1: End of November Progress Review

**What Professor Expects to See**:
```
✅ You have started working
✅ Some initial results (even if not good)
✅ Understanding of the task
✅ Problems identified
✅ Plan for next month

Questions Professor May Ask:
1. What approach are you using?
2. What problems have you encountered?
3. What have you learned so far?
4. What do you need help with?
5. What are your next steps?
```

**Your Team's Preparation**:
```
For Teammate (Subtask 1):
- Show baseline results
- Explain approach
- Discuss challenges
- Present learning progress

For You (Subtask 2a):
- Present completed models ✅
- Show ensemble results ✅
- Explain architecture decisions
- Discuss what you learned
- Demonstrate understanding of why things work
```

#### Milestone 2: Test Data Release & Submission

**Expected: Mid-December to January 9**

**Preparation Checklist**:
```
Before Test Data Release:
□ Understand submission format exactly
□ Test prediction pipeline on validation set
□ Verify CSV format matches requirements
□ Have ensemble script ready to run
□ Know how to create submission.zip

When Test Data Releases:
□ Download immediately
□ Verify data format
□ Run predictions (each subtask)
□ Generate CSV files
□ Validate format
□ Create submission.zip
□ Submit to Codabench
□ Verify submission received

After Submission:
□ Save all results
□ Screenshot leaderboard
□ Document final performance
□ Begin final analysis
```

#### Milestone 3: Final Report

**Professor's Requirements** (Likely):
```
Academic Project Report

Required Sections:
1. Executive Summary
2. Introduction & Background
3. Task Description
4. Methodology (detailed)
5. Experiments & Results
6. Analysis & Discussion
7. Individual Contributions ⭐
8. Learning Reflection ⭐
9. Conclusion
10. Code Appendix
11. References

Length: 10-15 pages
Format: Academic report (IEEE or ACL style)
Due: Likely late January / early February

Individual Contributions Section is CRITICAL:
- Clear delineation of who did what
- Evidence of individual work
- Reflection on personal growth
```

---

## 🆘 Support & Resources

### Professor Support

**Availability**:
```
✅ Remote meetings available (email to schedule)
✅ Regular office hours (check syllabus)
✅ Email questions anytime
✅ Progress reviews (end of November)
```

**How to Request Meeting**:
```
Email Template:

Subject: Meeting Request - SemEval Task 2 Progress

Dear Professor [Name],

Our team ([Team Name]) is working on SemEval 2026 Task 2
(Subtasks 1 and 2a). We would like to schedule a meeting
to discuss [specific topics/problems].

Current Status:
- [Brief progress summary]
- [Specific problems we're facing]

Could we schedule a 30-minute meeting during [suggest times]?

We can meet via [Zoom/Teams/in-person].

Thank you,
[Your Names]
```

**What to Bring to Meetings**:
```
✅ Current results (numbers/graphs)
✅ Code snippets (if discussing bugs)
✅ Specific questions (not "help us with everything")
✅ What you've tried already

❌ "We haven't started yet"
❌ "We don't know what to do"
❌ Expecting professor to write code
```

---

### PhD Student Support

**Available Support**:
```
✅ Technical debugging
✅ Code reviews
✅ Architecture advice
✅ Paper writing tips
✅ Experimental design
```

**Noyes Team** (mentioned by professor):
- Experienced in similar research areas
- Can provide guidance
- Contact via professor introduction

---

### Lab Resources

**Server Cluster Access**:
```
Available: GPU servers in lab
How to Get Access: Email professor to request
Use Case: Training larger models, longer experiments

Benefits:
✅ Better GPUs than Colab free tier
✅ No session timeouts
✅ More storage
✅ Multiple experiments in parallel

Your Current Status:
- Using: Google Colab free tier (T4 GPU)
- Sufficient for: Current models
- Consider upgrade if: Want to try larger models (RoBERTa-large, etc.)
```

---

### Online Resources

**Recommended by Professor**:
```
1. Course Materials
   - Deep learning lectures (your course)
   - Practical NLP course (recommended elective)

2. SemEval Resources
   - Task website: https://semeval2026task2.github.io/
   - Competition page: https://www.codabench.org/competitions/9963/
   - Mailing list: [subscribe via task website]

3. Papers
   - Previous SemEval papers (ACL Anthology)
   - Emotion prediction literature
   - Transformer architectures

4. Code Examples
   - Hugging Face documentation
   - PyTorch tutorials
   - Previous SemEval systems (GitHub)
```

**Additional Materials** (Professor mentioned):
> "I will also give some references and some additional materials
> that you can go and refer to if you would like to learn deeply
> about some specific things."

---

## 📊 Our Team Status

### Team Information

```
Team Name: [Your Team Name]
Members: 2
Task: SemEval 2026 Task 2
Participation Level: General (not competition-focused)

Member 1: [Teammate Name]
- Subtask: Subtask 1 (Longitudinal Affect Assessment)
- Status: In progress
- Background: [Skill level]

Member 2: [Your Name]
- Subtask: Subtask 2a (State Change Forecasting)
- Status: ✅ Training complete, awaiting test data
- Background: [Your skill level honestly stated]
```

---

### Current Progress Summary

#### Subtask 2a (Your Work)

**Status**: ✅ **95% Complete** (Awaiting test data only)

**Completed**:
```
✅ Data exploration and preprocessing
✅ Feature engineering (39 features)
✅ Model architecture design
   - RoBERTa-base encoder
   - BiLSTM (256 hidden, 2 layers)
   - Multi-head attention (8 heads)
   - Dual-head output
✅ Loss function optimization
   - Valence: 65% CCC + 35% MSE
   - Arousal: 70% CCC + 30% MSE
✅ Training 3 models (different seeds)
   - seed42: CCC 0.5053
   - seed123: CCC 0.5330
   - seed777: CCC 0.6554
✅ Ensemble system
   - Performance-based weighting
   - Expected: CCC 0.5846-0.6046
✅ Extensive documentation
   - 5 markdown documents
   - Complete training guide
   - Architecture analysis
✅ Code organization
   - Clean folder structure
   - Modular code
   - Reproducible
```

**Remaining**:
```
⏳ Test data prediction (when data releases)
⏳ Submission file creation
⏳ Final results evaluation
⏳ Paper writing
```

**Learning Demonstrated**:
```
Starting Point (estimated):
- [Your honest assessment]

Progress Made:
- Learned transformer architectures
- Implemented LSTM + Attention
- Mastered PyTorch training loops
- Understood ensemble methods
- Developed systematic experimentation approach
- Gained experience with Google Colab GPU training

Skills Acquired:
- Deep learning model design
- Hyperparameter optimization
- Performance analysis (CCC metric)
- Scientific writing and documentation
- Reproducible research practices
```

---

### Risk Assessment

**Potential Risks** and **Mitigation**:

```
Risk 1: Test data released very late (Jan 8)
Impact: No time to fix problems
Mitigation:
✅ Prepare prediction pipeline now
✅ Test on validation set
✅ Have backup plan ready

Risk 2: Teammate falling behind
Impact: Incomplete Subtask 1
Mitigation:
✅ Regular check-ins
✅ Offer to help if needed
✅ Be prepared to submit 2a only if necessary

Risk 3: Poor test set performance
Impact: Results different from validation
Mitigation:
✅ Not a problem for professor's evaluation!
✅ Focus on process and learning
✅ Analyze what went wrong (good for report)

Risk 4: Time pressure in January
Impact: Rushed final report
Mitigation:
✅ Start paper draft in December
✅ Document as you go
✅ Outline report structure now
```

---

### Success Criteria

#### For Professor's Evaluation (What Matters)

```
✅ Individual contribution clearly demonstrated
✅ Learning progress well-documented
✅ Technical quality of implementation
✅ Depth of analysis and understanding
✅ Honest reflection on process
✅ Good teamwork and collaboration

❌ Not important: Competition ranking
❌ Not important: Beating other teams
❌ Not important: Perfect results
```

#### For Personal Growth

```
✅ Deep understanding of emotion prediction
✅ Practical deep learning skills
✅ Research methodology experience
✅ Scientific writing practice
✅ Team collaboration experience
✅ Problem-solving skills
```

---

### Next Action Items

#### Immediate (This Week)

**For You (Subtask 2a)**:
```
□ Prepare prediction script for test data
□ Create submission format template
□ Start paper draft (outline)
□ Organize experiment logs
□ Document learning journey
□ Prepare for November progress meeting
```

**For Teammate (Subtask 1)**:
```
□ [Teammate's action items]
□ Share progress with you
□ Prepare for progress meeting
```

**For Team**:
```
□ Schedule weekly meeting time
□ Set up shared documentation
□ Prepare November progress presentation
□ Outline final report structure
```

#### December

```
□ Complete Subtask 1 experiments
□ Test prediction pipelines
□ Write paper draft
□ Conduct final ablations
□ Prepare submission materials
```

#### January

```
□ Submit predictions (by Jan 9)
□ Analyze final results
□ Complete paper
□ Write final report for professor
□ Prepare presentation (if required)
```

---

## 📖 Common Questions

### Q: Do we need to compete for ranking?

**A: No, general participation is fine.**

Professor evaluates based on learning and contribution, not ranking. Competition ranking is bonus, not requirement.

---

### Q: What if our results are not good?

**A: Show good process and learning.**

Professor said: "I will evaluate based on your current expertise and what you learn." Bad results with good process and analysis = Good grade.

---

### Q: Can we use existing code/models?

**A: Yes, but must clearly state what you reused vs created.**

Document:
- What you used (libraries, pretrained models)
- What you built on top
- Your original contributions

---

### Q: What if we encounter problems we can't solve?

**A: Document the problem and what you tried.**

Professor wants to see:
1. You identified the problem
2. You tried solutions
3. You learned from failures
4. You asked for help appropriately

This is BETTER than hiding problems.

---

### Q: How much code documentation is enough?

**A: More is better.**

Include:
- Code comments explaining key decisions
- Docstrings for classes/functions
- README files in each directory
- High-level architecture document
- Experiment logs

---

### Q: Do we both need to understand all the code?

**A: Understand yours deeply, others' at high level.**

You should:
- ✅ Explain your code in detail
- ✅ Understand teammate's approach conceptually
- ✅ Be able to discuss why team made certain decisions
- ❌ Not required to debug each other's implementation details

---

## 🎯 Final Reminders

### Most Important Points

1. **Be Honest** about your starting skill level
2. **Document Everything** - your work, learning, problems
3. **Individual Contribution** matters most for grade
4. **Progress and Growth** are evaluated, not just results
5. **Process Quality** is as important as final performance
6. **Teamwork** and collaboration should be demonstrated
7. **Ask for Help** when needed - it's expected and encouraged

### Professor's Philosophy

> "The final objective is not to get a high rank in a SemEval.
> Those who like, I'm really happy if you could go for high rank.
> And if you can't go, that's again fine. Because as I said earlier,
> I will evaluate your project outcomes based on your current
> expertise and what you learn and you are overcoming in a few
> months."

**Translation**:
- Focus on learning, not winning
- Show growth from your baseline
- Demonstrate good process
- Be honest about capabilities

---

# PART 3: Subtask 2a Overview

## 📊 Project Status

### ✅ Completed

**Subtask 2A**:
- ✅ 3 ensemble models trained (seeds: 42, 123, 777)
- ✅ Ensemble weights calculated
- ✅ Expected CCC: 0.5846-0.6046
- ✅ Documentation complete
- ✅ Code organized and clean

**Models**:
```
models/
├── subtask2a_seed42_best.pt   (CCC 0.5053, 1.5 GB)
├── subtask2a_seed123_best.pt  (CCC 0.5330, 1.5 GB)
└── subtask2a_seed777_best.pt  (CCC 0.6554, 1.5 GB)
```

**Results**:
```
results/subtask2a/
└── ensemble_results.json
```

---

### ⚠️ In Progress / Needed

**Subtask 2A - Final Steps**:
1. ❌ Obtain test dataset (`test_subtask2a.csv`)
2. ❌ Create prediction script for test data
3. ❌ Generate `pred_subtask2a.csv` with ensemble
4. ❌ Create `submission.zip`
5. ❌ Submit to Codabench
6. ❌ Write system description paper

**Subtask 1**:
- ⚠️ Baseline exists but not advanced model
- ❓ Decision needed: Submit or skip?

**Subtask 2B**:
- ❌ Completely not started
- ❓ Decision needed: Implement or skip?

---

### 🎯 Next Steps (Prioritized)

#### Option 1: Focus on Subtask 2A (Recommended)

**Why**: Nearly complete, highest chance of good results

**Steps**:
1. Wait for test data release
2. Create prediction script using ensemble
3. Generate submission file
4. Submit and evaluate
5. Write paper with strong analysis

**Timeline**: 1-2 days once test data available

---

#### Option 2: Add Subtask 2B

**Why**: Similar to 2A, can reuse most code

**Steps**:
1. Copy Subtask 2A code
2. Modify target to dispositional change
3. Train 3 models
4. Create ensemble
5. Submit both 2A and 2B

**Timeline**: 3-4 days additional work

---

#### Option 3: Improve Subtask 1

**Why**: Different from Task 2, diversifies submission

**Steps**:
1. Apply Subtask 2A architecture to Subtask 1
2. Train models
3. Submit all 3 subtasks

**Timeline**: 2-3 days additional work

---

## 📋 Checklist

### Before Submission

- [ ] Test data obtained
- [ ] Predictions generated in correct format
- [ ] CSV files have exact required filenames
- [ ] Column order matches requirements exactly
- [ ] Header row included
- [ ] All predictions within valid ranges
- [ ] ZIP file created correctly
- [ ] Tested submission file structure
- [ ] Registered on Codabench
- [ ] Submitted before deadline

### Before Paper Submission

- [ ] Title follows SemEval format
- [ ] Author names included (NOT anonymous)
- [ ] Length within limits (5 pages for single task)
- [ ] Task description paper cited
- [ ] Code/data URLs included (if releasing)
- [ ] Strong analysis section included
- [ ] Reproducibility details provided
- [ ] Ethical considerations discussed (if applicable)
- [ ] Languages mentioned
- [ ] Style files used correctly

---

**Document Version**: 1.0
**Last Updated**: November 23, 2025
**Integration of**: SEMEVAL_2026_TASK2_REQUIREMENTS.md + PROFESSOR_EVALUATION_GUIDE.md + subtask2a/README.md
**Purpose**: Unified project overview for reference and evaluation preparation

---

**Good luck with your project! 🎓**
