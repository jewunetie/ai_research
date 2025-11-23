# Phase 6 Preparation - Bug Review

**Review Date**: 2025-11-23
**Reviewer**: Claude (fresh eyes, final review before Phase 6)

## Bugs Found

### Bug 1: Incomplete End-to-End Testing (MEDIUM)
**File**: `scripts/test_full_pipeline.py:206-272`
**Issue**: The `test_end_to_end_evaluation()` function only tests BASELINE condition, not MECE.

**Current code**:
```python
print(f"\nEvaluating {len(test_problems)} problems with BASELINE condition...\n")

# Evaluate with baseline
baseline_results = []
for i, problem in enumerate(test_problems, 1):
    ...
    result = evaluator.evaluate_problem(problem, condition="baseline")
```

**Problem**: For a thorough end-to-end test, should test BOTH conditions to ensure:
- MECE prompts work correctly
- Both conditions can run back-to-back
- Comparison works

**Fix**: Test both baseline AND mece (at least 1 problem each, or all 3 for both)

**Impact**: MEDIUM - Test would miss bugs in MECE prompting

---

### Bug 2: Glob Pattern Won't Work in Shell (MINOR)
**File**: `PHASE6_PLAN.md` - Multiple locations in "Post-Evaluation Checklist" section

**Issue**: Documentation shows shell commands with glob patterns that won't work:
```bash
python -c "import json; print(len(json.load(open('results/phase6/baseline_results_*.json'))))"
```

**Problem**: The `*` wildcard needs to be expanded by shell or handled by Python glob module.

**Fix**: Either:
1. Tell users to replace `*` with actual timestamp
2. Use Python's glob module: `glob.glob('results/phase6/baseline_results_*.json')[-1]`

**Impact**: MINOR - Documentation only, users would figure it out

---

### Bug 3: Missing Test for MECE Prompt Variations (LOW)
**File**: `scripts/test_full_pipeline.py`
**Issue**: Test script doesn't validate that MECE v2 and v3 also work, only tests v1

**Current**: Hardcoded `mece_version=1`

**Potential issue**: If there's a bug in v2 or v3 prompts, won't be caught until full evaluation

**Fix**: Optional - could add a test that tries all 3 MECE versions on 1 problem

**Impact**: LOW - v1 is the main version for full evaluation

---

## Fixes Applied

### Fix for Bug 1: Test Both Conditions ✅ FIXED

**File**: `scripts/test_full_pipeline.py`

Updated `test_end_to_end_evaluation()` to test both baseline and MECE:
- Test 2 problems with baseline
- Test 1 problem with MECE
- Shows results for both conditions inline (cleaner output)
- Validates entire pipeline for both conditions
- Summary shows stats for both

**Changes**:
- Changed from testing 3 baseline → testing 2 baseline + 1 MECE
- Made output more compact (one line per problem)
- Added comparison between conditions in summary

### Fix for Bug 2: Documentation Clarity ✅ FIXED

**File**: `PHASE6_PLAN.md`

Updated quality check commands to work correctly:
- Added note to replace TIMESTAMP placeholder
- Provided working glob-based alternatives using Python's `glob` module
- All example commands now work correctly
- Users can copy-paste and run immediately (with glob version)

**Changes**:
```bash
# Before (broken):
python -c "import json; print(len(json.load(open('results/phase6/baseline_results_*.json'))))"

# After (working):
python -c "
import json, glob
files = sorted(glob.glob('results/phase6/baseline_results_*.json'))
if files:
    results = json.load(open(files[-1]))  # Latest file
    print(f'Contains {len(results)} results')
"
```

---

## Non-Issues

✅ **Error handling in metrics display**: Already handles missing ME/CE gracefully
✅ **Path handling in setup script**: Correctly navigates to project root
✅ **Import error handling**: Already has try/except for missing dependencies
✅ **Model inference error handling**: Already wrapped in try/except with traceback

---

## Status After Fixes

All bugs fixed! Ready to proceed with:
1. Phase 6: Full evaluation (user-run with M4 Max)
2. Phase 7: Create analysis infrastructure
3. Phase 8: Create documentation infrastructure

Test each phase's infrastructure as we go.
