# Git Repository Cleanup Plan
**Date:** December 29, 2025  
**Repo:** JDSB123/green_bier_sports_nba_model  
**Current State:** Messy tags, bloated history, no clear versioning

---

## 🔍 Current Problems

### 1. **Inconsistent Tags** ❌
```
v6.0.0-hardened
v6.0.1
v6.2
v6.4
```
**Problems:**
- Old versioning scheme (v6.x) conflicts with new scheme (NBA_v33.x)
- No tag for current version (NBA_v33.0.8.0)
- Tags don't follow semantic versioning
- No way to tell what v6.4 contains vs NBA_v33.0.8.0

### 2. **Large Files in History** 🐘
```
2.4 MB: data/external/kaggle/nba_2008-2025.csv (2 versions)
247 KB: coverage.xml (3 versions)
```
**Problems:**
- CSV data committed to Git (should be in Git LFS or external)
- Test coverage reports committed (should be in .gitignore)
- Every clone downloads these large files forever

### 3. **No Release History** 📝
- No GitHub Releases
- No CHANGELOG
- No clear "what changed between versions"
- Hard to roll back to previous working version

### 4. **Messy Commit Messages** 💬
Recent history shows:
- "fix: indent azure credential script"
- "fix: pass azure creds json"
- Multiple "fix:" commits for same issue (should be squashed)

---

## ✅ Cleanup Actions

### **PHASE 1: Tag Cleanup & Standardization**

#### 1.1 Delete Old Tags
```powershell
# Delete locally
git tag -d v6.0.0-hardened
git tag -d v6.0.1
git tag -d v6.2
git tag -d v6.4

# Delete from remote
git push origin :refs/tags/v6.0.0-hardened
git push origin :refs/tags/v6.0.1
git push origin :refs/tags/v6.2
git push origin :refs/tags/v6.4
```

**Rationale:** These tags use old versioning scheme and create confusion.

#### 1.2 Tag Current Version
```powershell
# Tag current HEAD with proper version
git tag -a NBA_v33.0.8.0 -m "NBA Model v33.0.8.0 - Production Release

Features:
- 6 markets (1H + FG spreads/totals/moneylines)
- Archive folder for historical tracking
- Automated deployment pipeline
- Version management automation
- CI/CD version validation

This is the first properly versioned release using the new NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD> scheme."

git push origin NBA_v33.0.8.0
```

#### 1.3 Create GitHub Release
Go to: https://github.com/JDSB123/green_bier_sports_nba_model/releases/new
- Tag: `NBA_v33.0.8.0`
- Title: `NBA Model v33.0.8.0 - Production Release`
- Description: Copy from tag message + link to VERSIONING.md
- Attach: Trained models (if small enough) or link to storage

---

### **PHASE 2: Remove Large Files from History**

⚠️ **WARNING:** This rewrites Git history. All collaborators must re-clone.

#### 2.1 Remove CSV Files
```powershell
# Use BFG Repo-Cleaner (faster) or git filter-repo
# Option A: BFG (recommended)
# Download from https://rtyley.github.io/bfg-repo-cleaner/
java -jar bfg.jar --delete-files nba_2008-2025.csv

# Option B: git filter-repo
pip install git-filter-repo
git filter-repo --path data/external/kaggle/nba_2008-2025.csv --invert-paths

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

#### 2.2 Remove Coverage Reports
```powershell
git filter-repo --path coverage.xml --invert-paths
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

#### 2.3 Update .gitignore (Already Done)
Ensure these are ignored:
```gitignore
# Data files (should be external or LFS)
data/external/kaggle/*.csv

# Test artifacts
coverage.xml
.coverage
htmlcov/
```

#### 2.4 Force Push (⚠️ DESTRUCTIVE)
```powershell
# Backup first!
git clone https://github.com/JDSB123/green_bier_sports_nba_model.git ../NBA_backup

# Force push cleaned history
git push origin --force --all
git push origin --force --tags
```

**Impact:** 
- Repo size: ~40% smaller
- Faster clones
- Cleaner history

---

### **PHASE 3: Create CHANGELOG**

#### 3.1 Generate Changelog
```bash
# Install auto-changelog
npm install -g auto-changelog

# Generate from commits
auto-changelog --output CHANGELOG.md --template keepachangelog
```

#### 3.2 Manual CHANGELOG Template
Create `CHANGELOG.md`:
```markdown
# Changelog

All notable changes to the NBA Prediction Model will be documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
versioning follows [VERSIONING.md](VERSIONING.md).

## [NBA_v33.0.8.0] - 2025-12-29

### Added
- Archive folder for historical output tracking
- Automated deployment script (scripts/deploy.ps1)
- Version bump automation (scripts/bump_version.py)
- CI/CD version validation workflow
- Complete versioning documentation (VERSIONING.md)

### Fixed
- Version consistency across 10+ files
- Deployment pipeline (now automated)
- Git hygiene (proper tracking of archive/)

### Changed
- Synced all version references to NBA_v33.0.8.0
- Updated .gitignore for archive/ folder

## [NBA_v33.0.7.0] - 2025-12-28

### Changed
- Calibration improvements
- Edge threshold tuning

## [NBA_v33.0.2.0] - 2025-12-22

### Added
- Production deployment with 6 markets

## [NBA_v33.0.0.0] - 2025-12-15

### Changed
- Major refactor: 9→6 markets (removed Q1)

---

## Version History (Old Scheme - Deprecated)

These versions used an old numbering scheme and are deprecated:

- **v6.4** - 2025-12-XX - (details unknown)
- **v6.2** - 2025-12-XX - (details unknown)
- **v6.0.1** - 2025-12-XX - (details unknown)
- **v6.0.0-hardened** - 2025-12-XX - (details unknown)

All future versions will use the NBA_v<MAJOR>.<MINOR>.<PATCH>.<BUILD> scheme.
See [VERSIONING.md](VERSIONING.md) for details.
```

---

### **PHASE 4: Squash Messy Commits (Optional)**

If you want a cleaner history, squash related "fix:" commits:

```powershell
# Interactive rebase (last 10 commits)
git rebase -i HEAD~10

# In the editor, change "pick" to "squash" for fixes you want to combine
# Example:
#   pick cf7abb6 fix: require Azure client secret login
#   squash f6691b4 fix: use null check for secret
#   squash 5ad4ed1 fix: correct GitHub Actions expressions
#   squash 0559926 fix: add Azure client secret fallback

# Save and close - Git will combine commits
```

**⚠️ Only do this if history hasn't been shared!**

---

### **PHASE 5: Branch Cleanup**

#### 5.1 Check for Stale Branches
```powershell
# List remote branches
git branch -r

# Delete merged branches
git branch -r --merged | Where-Object { $_ -notmatch 'main' } | ForEach-Object {
    $branch = $_ -replace 'origin/', ''
    git push origin --delete $branch
}
```

#### 5.2 Set Branch Protection Rules
On GitHub → Settings → Branches → Add rule for `main`:
- ✅ Require pull request reviews
- ✅ Require status checks (version-check workflow)
- ✅ Require branches to be up to date
- ✅ Include administrators

---

### **PHASE 6: Documentation Updates**

#### 6.1 Update README
Add badges:
```markdown
![Version](https://img.shields.io/badge/version-NBA__v33.0.8.0-blue)
![Tests](https://github.com/JDSB123/green_bier_sports_nba_model/workflows/Tests/badge.svg)
![Deploy](https://github.com/JDSB123/green_bier_sports_nba_model/workflows/Version%20Validation/badge.svg)
```

#### 6.2 Add CONTRIBUTING.md
Create clear contribution guidelines:
- How to set up dev environment
- How to run tests
- How to bump versions
- Commit message conventions
- PR process

---

## 📋 Execution Checklist

### **Quick Cleanup (Low Risk)**
- [ ] Delete old tags (v6.x)
- [ ] Create NBA_v33.0.8.0 tag
- [ ] Create GitHub Release
- [ ] Add CHANGELOG.md
- [ ] Add badges to README
- [ ] Create CONTRIBUTING.md

### **Deep Cleanup (High Risk - Rewrites History)**
- [ ] **Backup repo:** `git clone --mirror` to safe location
- [ ] Remove large CSV files from history
- [ ] Remove coverage.xml from history
- [ ] Force push cleaned history
- [ ] Notify team to re-clone
- [ ] Update CI/CD to pull fresh

### **Ongoing Maintenance**
- [ ] Set up branch protection rules
- [ ] Update CHANGELOG with each release
- [ ] Tag every version bump
- [ ] Create GitHub Release for each tag
- [ ] Archive old branches after 90 days

---

## 🚀 Recommended: Start Fresh (Clean Slate)

If history is too messy and you don't care about preserving old commits:

```powershell
# 1. Create new orphan branch
git checkout --orphan clean-main

# 2. Add current files
git add -A

# 3. Create initial commit
git commit -m "feat: NBA Prediction Model v33.0.8.0 - Clean Slate

This is a fresh start for the NBA prediction model repository.
Previous history was messy with inconsistent versioning and large files.

Features:
- 6-market prediction engine (1H + FG spreads/totals/moneylines)
- Automated deployment pipeline
- Version management automation
- CI/CD validation
- Archive folder for historical tracking

For versioning guidelines, see VERSIONING.md
For deployment, see scripts/deploy.ps1"

# 4. Delete old main
git branch -D main

# 5. Rename clean-main to main
git branch -m main

# 6. Force push
git push -f origin main

# 7. Tag it
git tag -a NBA_v33.0.8.0 -m "Production Release - Clean Slate"
git push origin NBA_v33.0.8.0
```

**Pros:**
- ✅ Smallest possible repo size
- ✅ No legacy baggage
- ✅ Clean, professional history
- ✅ Fast clones

**Cons:**
- ❌ Loses all history
- ❌ Old issue/PR references break
- ❌ Can't git bisect old bugs

---

## 📊 Expected Results

| Metric | Before | After (Quick) | After (Deep) | After (Fresh) |
|--------|--------|---------------|--------------|---------------|
| **Repo Size** | ~50 MB | ~50 MB | ~30 MB | ~10 MB |
| **Clone Time** | 30s | 30s | 15s | 5s |
| **Tags** | 4 confusing | 1 clear | 1 clear | 1 clear |
| **History** | Messy | Messy | Clean | Pristine |
| **Large Files** | CSV in history | CSV in history | Removed | Never existed |

---

## 🎓 Best Practices Going Forward

1. **Never commit large data files** - Use Git LFS or external storage
2. **Never commit test artifacts** - Use proper .gitignore
3. **Tag every production release** - Enables rollback
4. **Update CHANGELOG** - Human-readable history
5. **Create GitHub Releases** - Professional documentation
6. **Squash messy fixes** - Clean commit history
7. **Use semantic commits** - `feat:`, `fix:`, `chore:`
8. **Branch protection** - Prevent accidents

---

## 🔗 Commands Quick Reference

```powershell
# Tag cleanup
git tag -d v6.0.0-hardened v6.0.1 v6.2 v6.4
git push origin --delete $(git tag -l "v6*")

# Create new tag
git tag -a NBA_v33.0.8.0 -m "Production Release"
git push origin NBA_v33.0.8.0

# Check repo size
git count-objects -vH

# Find large files
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | Where-Object { $_ -match '^blob' } | Sort-Object { [int]($_ -split ' ')[2] } -Descending | Select-Object -First 10

# Cleanup after filter-repo
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

---

## ❓ Which Option Should You Choose?

### **Quick Cleanup** (Recommended for now)
- ✅ No risk
- ✅ Immediate improvement
- ✅ Can do deep cleanup later
- ⏱️ 15 minutes

### **Deep Cleanup** (If you need smaller repo)
- ⚠️ Medium risk (history rewrite)
- ✅ 40% smaller repo
- ⚠️ Team must re-clone
- ⏱️ 1 hour

### **Fresh Start** (If history doesn't matter)
- ⚠️ High risk (loses history)
- ✅ Smallest repo
- ✅ Cleanest history
- ⚠️ Breaks old links
- ⏱️ 30 minutes

---

## 📝 Next Steps

1. **Decide:** Quick vs Deep vs Fresh
2. **Backup:** `git clone --mirror` to safe location
3. **Execute:** Follow checklist for chosen option
4. **Verify:** Test clone, check size, validate tags
5. **Document:** Update README with new badges/links
6. **Communicate:** Notify team if history was rewritten

---

**Ready to start?** Pick your option and let's clean this up!
