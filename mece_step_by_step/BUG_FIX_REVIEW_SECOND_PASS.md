# Bug Fix Review - Second Pass

Found **1 REGRESSION BUG** introduced during Bug 2 fix!

## New Bug Found

### Bug 7: Missing Lowercase in normalize_solution() (REGRESSION - MEDIUM)
**Severity**: MEDIUM - Introduced during Bug 2 fix
**Introduced by**: Consolidating duplicate normalization implementations

**Issue**: When I removed `CollectiveExhaustivenessScorer._normalize_solution()` and consolidated to use `parsers.normalize_solution()`, I lost the `.lower()` call.

**Old CE code** (removed):
```python
def _normalize_solution(self, solution: str) -> str:
    normalized = re.sub(r'\s+', ' ', solution.strip())
    normalized = re.sub(r'([a-zA-Z])\s*=\s*', r'\1 = ', normalized)
    normalized = normalized.lower()  # ← Had lowercasing!
    return normalized
```

**Current parsers code**:
```python
def normalize_solution(solution: str) -> str:
    solution = re.sub(r'\s+', ' ', solution.strip())
    solution = re.sub(r'([a-zA-Z])=', r'\1 = ', solution)
    solution = re.sub(r'=([0-9])', r'= \1', solution)
    return solution.strip()  # ← Missing .lower()!
```

**Impact**:
- Before fix: CE would match "X = 3" against "x = 3" (case-insensitive)
- After fix: Neither CE nor Accuracy match "X = 3" against "x = 3" (case-sensitive)
- If model outputs uppercase variable names, they won't match ground truth
- Ground truth uses lowercase: `"x = 3"`, `"x = -2"`, etc.

**Why tests didn't catch it**:
- `test_phase3.py` only tests that normalized string contains 'x' and '3', not case matching
- `test_metrics.py` has proper case test but requires numpy (not installed yet)

**Fix needed**: Add `.lower()` to `parsers.normalize_solution()`

## Other Checks

✅ **Type hints**: All correct, using `Tuple` from typing
✅ **Imports**: `normalize_solution` properly imported in CE
✅ **Import placement**: `import re` at module level
✅ **Method removal**: `_normalize_solution()` correctly removed from CE
✅ **Function calls**: All 3 calls updated to use `normalize_solution()`

## Action Taken

✅ **Bug 7 FIXED**: Added `.lower()` to `parsers.normalize_solution()` at line 284

**Fix details**:
```python
# Before:
return solution.strip()

# After:
solution = solution.lower()
return solution.strip()
```

**Updated docstring** to document lowercase behavior with example.

**Test results**: All tests still pass ✅

**Benefit**: Both Accuracy and CE now use consistent case-insensitive matching!
