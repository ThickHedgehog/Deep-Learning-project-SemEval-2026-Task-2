# Team Workflow Guide

## 👥 Team Composition

- **Person A (Data & Model Expert)**
- **Person B (Temporal Modeling & System Expert)**

## 🔄 Development Workflow

### Git Branch Strategy

```
main (production-ready code)
├── develop (integration branch)
│   ├── feature/person-a/data-preprocessing
│   ├── feature/person-a/transformer-models
│   ├── feature/person-b/temporal-models
│   └── feature/person-b/evaluation-system
```

### Branch Naming Convention
- `feature/person-a/feature-name`: Person A's feature development
- `feature/person-b/feature-name`: Person B's feature development
- `hotfix/issue-description`: Emergency fixes
- `release/v1.0.0`: Release preparation

### Commit Message Rules
```
feat: Add new feature
fix: Bug fix
docs: Documentation update
style: Code formatting
refactor: Code refactoring
test: Add/modify tests
chore: Other tasks

Examples:
feat(models): Add temporal LSTM emotion model
fix(data): Fix preprocessing pipeline memory leak
docs(readme): Update installation instructions
```

## 📅 Weekly Schedule

### Every Monday (Planning Meeting)
- **Time**: 10:00 AM - 11:00 AM
- **Purpose**: Set weekly goals and distribute tasks
- **Agenda**:
  - Review last week's progress
  - Set this week's goals
  - Determine task priorities
  - Discuss blockers and dependencies

### Every Wednesday (Mid-week Check-in)
- **Time**: 3:00 PM - 3:30 PM
- **Purpose**: Check progress and resolve issues
- **Agenda**:
  - Share individual progress
  - Discuss encountered problems
  - Adjust tasks if needed

### Every Friday (Code Review & Integration)
- **Time**: 4:00 PM - 5:00 PM
- **Purpose**: Code review and weekly summary
- **Agenda**:
  - Review Pull Requests
  - Code integration and testing
  - Summarize weekly achievements
  - Prepare for next week

## 🔧 Development Environment Setup

### Common Development Environment
```bash
# Python version: 3.8+
# Virtual environment setup
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Setup pre-commit hooks
pre-commit install
```

### Code Quality Tools
```bash
# Code formatting
black src/ tests/ scripts/

# Linting
flake8 src/ tests/ scripts/

# Type checking
mypy src/

# Run tests
pytest tests/

# Check coverage
pytest --cov=src tests/
```

## 📋 Task Management

### GitHub Issues Labels
- `priority-high`: High priority
- `priority-medium`: Medium priority
- `priority-low`: Low priority
- `person-a`: Person A's responsibility
- `person-b`: Person B's responsibility
- `bug`: Bug fix
- `feature`: New feature
- `documentation`: Documentation work
- `research`: Research work

### Task Board (GitHub Projects)
```
To Do → In Progress → Review → Done

Create each task as an Issue with appropriate labels
Regularly update the board for progress tracking
```

## 🤝 Collaboration Rules

### Pull Request Rules
1. **Create Branch**: `feature/person-{a|b}/feature-name`
2. **Create PR after completing work**
3. **PR Title**: Clear and concise description
4. **PR Description**: Changes, test results, related issue links
5. **Assign Reviewer**: Assign the other person as reviewer
6. **Merge after Approval**: Minimum 1 approval required

### Code Review Guidelines
- **Constructive Feedback**: Focus on improvement suggestions
- **Clear Explanations**: Explain why changes are needed
- **Quick Response**: Complete reviews within 24 hours
- **Prefer Small PRs**: Split large changes into multiple PRs

### Communication Tools
- **Slack/Discord**: Daily communication
- **GitHub Issues**: Official work discussions
- **Google Docs**: Collaborative document work
- **Zoom/Meet**: Meetings and pair programming

## 📊 Experiment Management

### WandB Project Setup
```python
import wandb

# Initialize project
wandb.init(
    project="semeval-2026-task2",
    name=f"experiment-{person}-{date}",
    tags=["baseline", "bert", "temporal"]
)
```

### Experiment Naming Convention
```
{person}_{model_type}_{date}_{description}

Examples:
person_a_bert_baseline_20250128_initial
person_b_lstm_temporal_20250128_bidirectional
```

### Model Version Management
```
models/
├── checkpoints/
│   ├── person_a/
│   │   ├── bert_baseline_v1.0/
│   │   └── transformer_ensemble_v2.1/
│   └── person_b/
│       ├── lstm_temporal_v1.0/
│       └── hierarchical_model_v1.5/
└── final/
    ├── best_single_model.pt
    └── best_ensemble_model.pt
```

## 🎯 Monthly Goals & Role Distribution

### Month 1 (Week 1-4)
**Person A**:
- [ ] Complete data exploration and analysis
- [ ] Build basic preprocessing pipeline
- [ ] Implement BERT baseline model
- [ ] Implement initial evaluation metrics

**Person B**:
- [ ] Project structure and environment setup
- [ ] Implement basic LSTM temporal model
- [ ] Build experiment tracking system
- [ ] Set up code quality tools

**Joint Work**:
- [ ] Weekly paper review sessions
- [ ] Establish code review process
- [ ] Initial baseline performance evaluation

### Month 2 (Week 5-8)
**Person A**:
- [ ] Implement advanced Transformer models
- [ ] Feature engineering and data augmentation
- [ ] Develop multi-scale models
- [ ] Model ensemble framework

**Person B**:
- [ ] Advanced temporal modeling techniques
- [ ] Implement hierarchical temporal models
- [ ] Optimize attention mechanisms
- [ ] Advanced evaluation system

**Joint Work**:
- [ ] Model performance comparison analysis
- [ ] Hyperparameter tuning
- [ ] Mid-term results summary

### Month 3 (Week 9-12)
**Person A**:
- [ ] Latest LLM utilization research
- [ ] Model compression and optimization
- [ ] Implement advanced ensemble techniques
- [ ] Performance benchmarking

**Person B**:
- [ ] Innovative temporal modeling
- [ ] Loss function optimization
- [ ] Automated hyperparameter tuning
- [ ] System integration and optimization

**Joint Work**:
- [ ] Select best performing models
- [ ] Cross-validation and robustness testing
- [ ] Summarize research results

### Month 4 (Week 13-16)
**Person A**:
- [ ] Final model validation and testing
- [ ] Generate and format prediction results
- [ ] Ensure reproducibility and clean code
- [ ] Prepare submission files

**Person B**:
- [ ] Write system description
- [ ] Code documentation and API docs
- [ ] Write research paper
- [ ] Prepare final presentation materials

**Joint Work**:
- [ ] Complete Codabench submission
- [ ] Write final report
- [ ] Project retrospective and summary

## 🚨 Crisis Management

### Technical Issues
- **Model Performance Issues**: Immediately share with teammate, joint debugging
- **Data Problems**: Backup strategies and alternative data sources
- **Computational Resource Shortage**: Utilize cloud resources and model compression

### Schedule Delays
- **Priority Reordering**: Focus on core functionality first
- **Task Redistribution**: Adjust roles if necessary
- **External Help**: Mentoring or external resources

### Communication Issues
- **Regular Check-ins**: Early problem detection
- **Clear Documentation**: Record all decisions
- **Open Communication**: Share difficulties immediately

## 📈 Performance Measurement

### Weekly KPIs
- Number of code commits
- Number of Pull Requests created/reviewed
- Number of issues resolved
- Number of experiments completed

### Monthly Evaluation
- Goal achievement rate
- Model performance improvement
- Code quality metrics
- Documentation completeness

### Final Performance Indicators
- **Competition Ranking**: Top 10 target
- **Technical Contribution**: New approach development
- **Code Quality**: 90%+ test coverage
- **Documentation**: Complete reproducibility

---

This workflow enables efficient and systematic collaboration. Regularly update this document to adjust to the team's situation.