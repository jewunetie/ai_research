# Critical Bugs Found in Phase 8 Implementation

## CRITICAL BUGS

### 1. **NameError in analyze_main.py (Line 205)**

**Location**: `experiments/main/analyze_main.py:205`

**Bug**: Variable `results_file` is not in scope

**Current Code**:
```python
def print_analysis(results: Dict):
    """Print comprehensive analysis of results."""
    # ... analysis code ...

    # Save DataFrame for further analysis
    output_dir = Path(results_file).parent  # ❌ results_file not defined!
    df_file = output_dir / "analysis_dataframe.csv"
```

**Problem**: `results_file` is only available in `main()` function scope, not in `print_analysis()`. This will cause a `NameError` when trying to save the CSV file.

**Fix**: Need to pass `results_file` as parameter or derive output path from results data.

---

### 2. **Redundant If-Elif Chain in run_main.py (Lines 339-344)**

**Location**: `experiments/main/run_main.py:339-344`

**Bug**: All branches do exactly the same thing

**Current Code**:
```python
# Get baseline context
if baseline_name == "full_context":
    context, metadata = baseline.process(text)
elif baseline_name == "no_context":
    context, metadata = baseline.process(text)
elif baseline_name == "random_tokens":
    context, metadata = baseline.process(text)
```

**Problem**: Unnecessary complexity - all three branches call the same method.

**Should be**:
```python
# Get baseline context
context, metadata = baseline.process(text)
```

This is not a critical bug (code works) but indicates poor code quality.

---

### 3. **KeyError Risk in run_main.py (Line 296)**

**Location**: `experiments/main/run_main.py:296`

**Bug**: No error handling for missing prompt keys

**Current Code**:
```python
for variant in self.config.compression_variants:
    prompt_template = PROMPT_REGISTRY[variant.prompt_key]  # ❌ Could KeyError
```

**Problem**: If YAML config contains an invalid `prompt_key`, this will crash with `KeyError`.

**Fix**: Should validate prompt_key exists or use `.get()` with error handling:
```python
prompt_template = PROMPT_REGISTRY.get(variant.prompt_key)
if prompt_template is None:
    raise ValueError(f"Unknown prompt_key: {variant.prompt_key}")
```

---

### 4. **Inaccurate Cost Tracking (Line 281)**

**Location**: `experiments/main/run_main.py:281`

**Bug**: Cost tracking uses placeholder instead of actual prompt

**Current Code**:
```python
# Step 1: Generate questions (once per document)
qa_prompt = f"Based on the following text, generate {self.config.questions_per_document} diverse questions..."
qa_pairs = self.supervisor.generate_questions(
    text,
    num_questions=self.config.questions_per_document
)
self._track_api_call(qa_prompt, str(qa_pairs))  # ❌ Wrong prompt used
```

**Problem**: `qa_prompt` is just a placeholder string, not the actual prompt sent to the API. The real prompt is constructed inside `supervisor.generate_questions()` using `QUESTION_GENERATION_PROMPT` template.

**Impact**: Cost tracking will underestimate token usage for question generation.

**Fix**: Need to either:
1. Return actual prompt from supervisor
2. Track API calls inside the supervisor/answerer methods
3. Build the full prompt here for tracking

---

### 5. **Division by Zero Risk (Line 391)**

**Location**: `experiments/main/run_main.py:391`

**Bug**: No check for zero documents before division

**Current Code**:
```python
"summary": {
    "total_attempted": self.config.num_documents,
    "successful": len(self.results),
    "failed": len(self.failed_docs),
    "completion_rate": len(self.results) / self.config.num_documents,  # ❌ Could be 0/0
}
```

**Problem**: If `num_documents` is 0 (edge case), this causes `ZeroDivisionError`.

**Fix**:
```python
"completion_rate": len(self.results) / self.config.num_documents if self.config.num_documents > 0 else 0.0,
```

---

### 6. **Division by Zero in _print_summary (Lines 415, 440, 456)**

**Location**: `experiments/main/run_main.py:415, 440, 456`

**Bug**: No check before computing averages

**Current Code**:
```python
print(f"Success rate: {len(self.results)/self.config.num_documents*100:.1f}%")  # Line 415

# Line 440
print(f"    F1: {sum(all_f1)/len(all_f1):.3f}")

# Line 456
print(f"    F1: {sum(all_f1)/len(all_f1):.3f}")
```

**Problem**:
- Line 415: Division by zero if num_documents is 0
- Lines 440, 456: Division by zero if all_f1 is empty (already have `if all_f1:` check, so this is OK)

**Fix for line 415**:
```python
success_rate = (len(self.results) / self.config.num_documents * 100) if self.config.num_documents > 0 else 0.0
print(f"Success rate: {success_rate:.1f}%")
```

---

## MEDIUM PRIORITY ISSUES

### 7. **Missing Validation for Empty Baselines**

**Location**: `experiments/main/run_main.py:122`

**Issue**: No warning if no baselines enabled

**Current Code**:
```python
print(f"✓ Baselines: {', '.join(self.baselines.keys())}")
```

**Problem**: If config has no baselines, this prints `"✓ Baselines: "` which might be confusing.

**Fix**: Add check:
```python
if self.baselines:
    print(f"✓ Baselines: {', '.join(self.baselines.keys())}")
else:
    print("⚠  No baselines enabled")
```

---

### 8. **Missing Validation for Empty Variants**

**Location**: `experiments/main/run_main.py:192`

**Issue**: No error if no compression variants specified

**Current Code**:
```python
print(f"Compression variants: {len(self.config.compression_variants)}")
```

**Problem**: Experiment will silently produce no compression results if variants list is empty.

**Fix**: Add validation:
```python
if len(self.config.compression_variants) == 0:
    raise ValueError("No compression variants specified in config")
```

---

## MINOR ISSUES

### 9. **Unused Config Fields**

**Location**: `src/utils/config.py:55-56`

**Issue**: Fields defined but never used

**Current Code**:
```python
save_compressions: bool = True
save_qa_details: bool = True
```

**Problem**: These flags are loaded from config but never checked in the code. The code always saves everything.

**Impact**: Misleading configuration options.

**Fix**: Either remove these fields or implement the filtering logic.

---

### 10. **Inconsistent Error Messages**

**Location**: Various

**Issue**: Some errors use "Error:", some use "✗", some use "❌"

**Examples**:
- Line 224: `"Error: Results file not found"`
- Line 260: `"✗ Error processing"`
- In verify_api.py: `"❌ ERROR:"`

**Impact**: Inconsistent user experience

**Fix**: Standardize on one format (suggest using "✗" consistently)

---

## SUMMARY

**Critical (Must Fix):**
1. ❌ NameError in analyze_main.py (will crash)
2. ❌ KeyError risk with invalid prompt_key (will crash)
3. ❌ Division by zero risk in summary (will crash on edge case)

**Important (Should Fix):**
4. ⚠️ Redundant if-elif chain (code smell)
5. ⚠️ Inaccurate cost tracking (wrong estimates)
6. ⚠️ Missing validation for empty variants (silent failure)

**Nice to Have:**
7. ℹ️ Missing baseline validation
8. ℹ️ Unused config fields
9. ℹ️ Inconsistent error messages

**Total Critical Bugs: 3**
**Total Issues Found: 10**
