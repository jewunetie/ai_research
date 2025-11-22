# Bugs and Issues Found in Implementation

## Critical Bugs

### 1. Division by Zero in pilot.py (Lines 244-252)

**Location**: `scripts/pilot.py:244-252`

**Issue**: If no documents are successfully processed, the `compressed_em`, `full_em`, `no_em`, `compressed_f1`, `full_f1`, `no_f1` lists will be empty, causing division by zero when computing averages.

**Current Code**:
```python
print("Average Exact Match:")
print(f"  Self-Compression:  {sum(compressed_em)/len(compressed_em):.3f}")
print(f"  Full Context:      {sum(full_em)/len(full_em):.3f}")
print(f"  No Context:        {sum(no_em)/len(no_em):.3f}")
```

**Fix**: Add check for empty lists before dividing.

---

### 2. Division by Zero in pilot.py (Line 257)

**Location**: `scripts/pilot.py:257`

**Issue**: If no documents are successfully processed, `compression_ratios` list will be empty, causing division by zero.

**Current Code**:
```python
compression_ratios = [doc["compression_metadata"]["compression_ratio"] for doc in results]
avg_ratio = sum(compression_ratios) / len(compression_ratios)
```

**Fix**: Add check for empty list before dividing.

---

## Medium Priority Issues

### 3. Missing scikit-learn Dependency

**Location**: `pyproject.toml`

**Issue**: `src/evaluation/metrics.py` line 114 imports `sklearn.metrics.pairwise.cosine_similarity`, but scikit-learn is not explicitly listed in dependencies. While it comes with sentence-transformers, it should be explicit.

**Fix**: Add `scikit-learn` to dependencies or add a more informative error message.

---

### 4. No Validation for Empty QA Pairs

**Location**: `scripts/pilot.py:120-136`

**Issue**: If `supervisor.generate_questions()` returns an empty list, the script continues without checking, leading to misleading results.

**Fix**: Add validation that `len(qa_pairs) > 0` before proceeding.

---

## Low Priority / Style Issues

### 5. Inconsistent Error Handling in Pilot Script

**Location**: `scripts/pilot.py:117-195`

**Issue**: When an exception occurs in the try block, it prints an error and continues, but this means the summary statistics might be computed on partial results without clear indication to the user how many documents failed.

**Suggestion**: Track and report number of failed documents in summary.

---

### 6. Missing Type Hints in Some Functions

**Location**: Various files

**Issue**: Some functions are missing return type hints (e.g., `NoContextBaseline.process` returns `tuple` instead of `Tuple[str, dict]`).

**Fix**: Add complete type hints for consistency.

---

### 7. Hard-coded Encoding Name

**Location**: `src/models/openai_model.py:59`

**Issue**: Falls back to `"o200k_base"` encoding, which is specific to GPT-4o/5 models. This might not work for other models.

**Mitigation**: This is acceptable given the target models, but worth noting.

---

## Non-Issues (Verified as Correct)

### ✓ Relative Imports

All imports within the `src/` package correctly use relative imports (e.g., `from ..models.base import BaseLLM`).

### ✓ Token Counter Implementation

Token counter properly uses token-based truncation via encoding, not character-based.

### ✓ API Fallback Logic

OpenAI model has proper fallback from Responses API to Chat Completions API with error handling.

### ✓ Metadata Tracking

All pipeline components properly track and return metadata.

---

## Recommendations

1. **Fix division by zero bugs immediately** (items #1 and #2)
2. Add input validation for empty QA pairs (item #4)
3. Add scikit-learn to dependencies explicitly
4. Consider adding comprehensive error logging
5. Add unit tests for edge cases (empty inputs, API failures, etc.)
