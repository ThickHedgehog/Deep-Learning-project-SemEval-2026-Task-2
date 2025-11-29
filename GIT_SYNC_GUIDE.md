# 🔄 Git 동기화 가이드 (PC ↔ 노트북)

## 현재 상태 확인 (2025-11-29)

### ✅ 이 PC (노트북) 상태
```
Branch: main
Remote: origin/main
Status: Up to date
Last Commit: 5989547 "Remove data/raw from Git tracking (keep locally only)"
```

**이 PC는 GitHub와 완전히 동기화되어 있습니다!**

---

## 📋 PC ↔ 노트북 동기화 방법

### 1️⃣ **노트북에서 작업 완료 후 GitHub로 푸시**

```bash
# 1. 변경사항 확인
git status

# 2. 변경된 파일 추가 (선택적으로)
git add <파일명>
# 또는 모든 변경사항 추가
git add .

# 3. 커밋
git commit -m "작업 내용 설명"

# 4. GitHub로 푸시
git push origin main
```

### 2️⃣ **다른 PC에서 최신 변경사항 가져오기**

```bash
# 1. 최신 변경사항 가져오기
git pull origin main

# 2. 충돌 발생 시 해결 후
git add .
git commit -m "Merge conflicts resolved"
git push origin main
```

---

## 🚨 일반적인 동기화 문제 해결

### 문제 1: "Your branch is behind 'origin/main'"

**원인**: GitHub에 더 최신 커밋이 있음

**해결**:
```bash
git pull origin main
```

### 문제 2: "Your branch is ahead of 'origin/main'"

**원인**: 로컬에 푸시하지 않은 커밋이 있음

**해결**:
```bash
git push origin main
```

### 문제 3: "Merge conflict" (충돌)

**원인**: 같은 파일을 두 PC에서 수정함

**해결**:
```bash
# 1. 충돌 파일 열기 (VSCode에서 자동 표시됨)
# 2. 충돌 부분 수정 (<<<<<<, ======, >>>>>> 표시 제거)
# 3. 수정 후
git add <충돌파일>
git commit -m "Resolve merge conflict"
git push origin main
```

### 문제 4: "fatal: Authentication failed"

**원인**: GitHub 인증 만료

**해결**:
```bash
# VSCode에서:
# 1. Ctrl+Shift+P
# 2. "Git: Clone" 검색
# 3. GitHub 계정 재로그인

# 또는 Personal Access Token 사용:
git remote set-url origin https://<TOKEN>@github.com/ThickHedgehog/Deep-Learning-project-SemEval-2026-Task-2.git
```

### 문제 5: "diverged branches" (브랜치 분기)

**원인**: 두 PC에서 각각 다른 커밋 생성

**해결**:
```bash
# 방법 1: Pull 후 자동 merge
git pull origin main --no-rebase

# 방법 2: Rebase (더 깔끔한 히스토리)
git pull origin main --rebase

# 충돌 발생 시
git add .
git rebase --continue
git push origin main
```

---

## 📁 로컬 전용 파일 관리 (.gitignore)

### 현재 제외된 폴더 (로컬에만 존재)
```
models/     (4.3 GB - 모델 파일)
docs/       (문서 파일)
data/raw/   (훈련 데이터)
```

**이 폴더들은 Git으로 동기화되지 않습니다!**

### 다른 PC로 옮기는 방법

#### 방법 1: 수동 복사 (USB/클라우드)
```bash
# 1. 압축
zip -r models.zip models/
zip -r docs.zip docs/
zip -r data.zip data/raw/

# 2. USB/Google Drive로 복사
# 3. 다른 PC에서 압축 해제
```

#### 방법 2: Git LFS 사용 (권장하지 않음 - 파일 크기 문제)
```bash
# .gitignore에서 해당 폴더 제거 후
git lfs track "models/*.pt"
git add .gitattributes models/
git commit -m "Add models to LFS"
git push origin main
```

#### 방법 3: Google Drive / OneDrive 동기화 폴더 사용
```
1. 프로젝트 폴더 전체를 클라우드 동기화 폴더로 이동
2. 다른 PC에서도 동일한 클라우드 폴더 사용
3. Git은 코드만, 클라우드는 큰 파일 동기화
```

---

## 🔍 VSCode에서 Git 상태 확인

### VSCode UI 사용
1. **Source Control 패널** (Ctrl+Shift+G)
   - Changes: 수정된 파일 목록
   - Staged Changes: 커밋 준비된 파일
   - Sync Changes: 푸시/풀 필요한 커밋 수

2. **하단 상태바**
   - 브랜치 이름 (main)
   - 화살표 ↓↑ (pull/push 필요한 커밋 수)

3. **Timeline 패널**
   - 파일별 커밋 히스토리 확인

---

## ✅ 동기화 전 체크리스트

### 작업 종료 시 (노트북 → GitHub)
- [ ] `git status` 실행 (변경사항 확인)
- [ ] `git add .` (모든 변경사항 추가)
- [ ] `git commit -m "작업 내용"` (커밋)
- [ ] `git push origin main` (푸시)

### 작업 시작 시 (다른 PC → GitHub)
- [ ] `git status` 실행 (현재 상태 확인)
- [ ] `git pull origin main` (최신 변경사항 가져오기)
- [ ] 충돌 확인 및 해결
- [ ] 작업 시작

---

## 🛠️ 유용한 Git 명령어

```bash
# 현재 상태 확인
git status

# 최근 커밋 히스토리
git log --oneline -10

# 원격 저장소 확인
git remote -v

# 브랜치 확인
git branch -vv

# 변경사항 비교
git diff

# 특정 파일 변경 취소
git checkout -- <파일명>

# 마지막 커밋 취소 (변경사항 유지)
git reset --soft HEAD~1

# 강제로 원격 브랜치와 동일하게 만들기 (주의!)
git reset --hard origin/main
```

---

## 🚀 권장 워크플로우

### 매일 작업 시작 시
```bash
git pull origin main
```

### 작업 중 (1-2시간마다)
```bash
git add .
git commit -m "WIP: 작업 중 설명"
git push origin main
```

### 작업 완료 시
```bash
git add .
git commit -m "완료: 작업 내용 상세 설명"
git push origin main
```

---

## 📞 추가 도움이 필요한 경우

### VSCode Git 설정
```bash
# Git 사용자 정보 확인
git config --list

# 사용자 이름 설정
git config --global user.name "Your Name"

# 이메일 설정
git config --global user.email "your.email@example.com"
```

### GitHub 인증 방법
1. **Personal Access Token** (권장)
   - GitHub Settings → Developer settings → Personal access tokens
   - repo 권한 부여
   - Token을 비밀번호로 사용

2. **SSH Key**
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   # 생성된 공개키를 GitHub에 등록
   ```

---

## 📊 현재 프로젝트 상태

```
Branch: main
Last Sync: 2025-11-29
Commits Ahead: 0
Commits Behind: 0
Status: ✅ Up to date
```

**현재 이 노트북은 GitHub와 완전히 동기화되어 있습니다!**

다른 PC에서 `git pull origin main`을 실행하면 동일한 상태가 됩니다.
