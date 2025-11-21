# Implementation Plan: MECE Step by Step Reasoning

**Target Hardware**: M4 Max MacBook Pro
**Model**: Qwen3-0.6B
**Focus**: Math Case Analysis Problems
**Comparison**: MECE Prompting vs Baseline CoT
**Date**: 2025-11-21

---

## 1. Scope and Constraints

### **Hardware-Optimized Design**
- **Platform**: M4 Max MacBook Pro (Apple Silicon)
- **Model**: Qwen3-0.6B (smallest Qwen3 model)
- **Framework**: MLX (Apple's optimized framework) or llama.cpp
- **Memory**: Efficient - 0.6B model fits comfortably in unified memory
- **Performance**: Fast inference expected on M4 Neural Engine

### **Experimental Scope**
- **Primary Comparison**: MECE prompting vs Baseline CoT (2 conditions)
- **Domain**: Math problems requiring case analysis
- **Problem Types**:
  - Absolute value equations (e.g., |x-3| = 5)
  - Piecewise functions
  - Sign analysis (positive/negative/zero cases)
  - Range-based problems (e.g., solve for x > 0, x < 0, x = 0)
- **Dataset Size**: Start with 50 problems (tractable for manual creation/validation)

### **Why This Scope Works**
1. **Verifiable**: Math case analysis has ground truth case structures
2. **Clear MECE Definition**: Cases must be mutually exclusive and collectively exhaustive
3. **Small Model Test**: If MECE helps 0.6B, it's a strong signal
4. **M4 Max Friendly**: Fast iteration with small model
5. **Focused**: Single domain, clear comparison

---

## 2. Computational MECE Metrics

### **2.1 Mutual Exclusivity (ME) Score**

**Definition**: Measures whether reasoning steps logically overlap

**Measurement Approach**:

1. **Embedding-Based Similarity**
   - Extract individual reasoning steps from model output
   - Compute embeddings for each step using sentence-transformers
   - Calculate pairwise cosine similarity
   - **ME Score** = 1 - (average pairwise similarity)
   - Higher score = more mutually exclusive

2. **Case Condition Overlap** (Math-Specific)
   - Parse case conditions from reasoning (e.g., "when x > 0", "when x < 0")
   - Check if conditions overlap (e.g., x ≥ 0 and x > 0 overlap)
   - **Overlap Count** = number of overlapping condition pairs
   - **ME Score** = 1 - (overlap_count / total_pairs)

**Implementation**:
```python
def compute_mutual_exclusivity(reasoning_steps: List[str]) -> Dict[str, float]:
    """
    Args:
        reasoning_steps: List of reasoning steps extracted from model output

    Returns:
        {
            'embedding_me_score': float,  # 0-1, higher is better
            'case_overlap_me_score': float,  # 0-1, higher is better
            'avg_pairwise_similarity': float,  # 0-1, lower is better
            'num_overlapping_cases': int
        }
    """
    # Embedding-based
    embeddings = sentence_transformer.encode(reasoning_steps)
    similarities = cosine_similarity_matrix(embeddings)
    avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])

    # Case overlap (math-specific)
    conditions = extract_case_conditions(reasoning_steps)
    overlaps = count_overlapping_conditions(conditions)

    return {
        'embedding_me_score': 1 - avg_similarity,
        'case_overlap_me_score': 1 - (overlaps / max(1, len(conditions))),
        'avg_pairwise_similarity': avg_similarity,
        'num_overlapping_cases': overlaps
    }
```

### **2.2 Collective Exhaustiveness (CE) Score**

**Definition**: Measures whether all necessary cases are covered

**Measurement Approach for Math Case Analysis**:

1. **Case Enumeration (Primary Method)**
   - Define ground truth cases for each problem type
   - Example for |x-3| = 5:
     - Case 1: x - 3 = 5 (when x - 3 ≥ 0)
     - Case 2: -(x - 3) = 5 (when x - 3 < 0)
     - Required cases: 2
   - Extract cases from model reasoning
   - **CE Score** = (covered_cases / required_cases)

2. **Condition Coverage**
   - For each problem, define required condition space
   - Example for sign analysis: must cover x > 0, x < 0, x = 0
   - Check if model reasoning covers all required conditions
   - **Coverage Score** = (covered_conditions / required_conditions)

**Implementation**:
```python
def compute_collective_exhaustiveness(
    reasoning_steps: List[str],
    problem: Dict[str, Any]
) -> Dict[str, float]:
    """
    Args:
        reasoning_steps: List of reasoning steps
        problem: Problem dict with 'type', 'required_cases', 'ground_truth'

    Returns:
        {
            'ce_score': float,  # 0-1, 1.0 means all cases covered
            'covered_cases': List[str],
            'missing_cases': List[str],
            'extra_cases': List[str]
        }
    """
    required_cases = problem['required_cases']
    covered_cases = extract_cases_from_reasoning(reasoning_steps)

    # Match covered cases to required cases
    matched = match_cases(covered_cases, required_cases)
    missing = [c for c in required_cases if c not in matched]
    extra = [c for c in covered_cases if c not in matched]

    ce_score = len(matched) / len(required_cases) if required_cases else 1.0

    return {
        'ce_score': ce_score,
        'covered_cases': matched,
        'missing_cases': missing,
        'extra_cases': extra
    }
```

### **2.3 Combined MECE Score**

```python
def compute_mece_score(me_score: float, ce_score: float) -> float:
    """Harmonic mean of ME and CE scores"""
    if me_score + ce_score == 0:
        return 0.0
    return 2 * (me_score * ce_score) / (me_score + ce_score)
```

### **2.4 Answer Accuracy**

```python
def compute_accuracy(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth
    Handles multiple solutions (e.g., x = 2 or x = -2)
    """
    pred_solutions = parse_solutions(predicted)
    gt_solutions = parse_solutions(ground_truth)
    return set(pred_solutions) == set(gt_solutions)
```

---

## 3. Dataset: Math Case Analysis Problems

### **3.1 Problem Categories**

**Category 1: Absolute Value Equations** (15 problems)
- Example: Solve |x - 3| = 5
- Required cases: x - 3 ≥ 0 and x - 3 < 0
- Ground truth: x = 8 or x = -2

**Category 2: Piecewise Functions** (15 problems)
- Example: Evaluate f(x) = {x^2 if x < 0, 2x if x ≥ 0} at x = -2, 0, 3
- Required cases: x < 0, x ≥ 0
- Ground truth: f(-2) = 4, f(0) = 0, f(3) = 6

**Category 3: Sign Analysis** (10 problems)
- Example: For what values of x is (x-1)(x+2) > 0?
- Required cases: x < -2, -2 < x < 1, x > 1
- Ground truth: x < -2 or x > 1

**Category 4: Range-Based Problems** (10 problems)
- Example: Find integer solutions to x^2 < 10
- Required cases: positive integers, negative integers, zero
- Ground truth: x ∈ {-3, -2, -1, 0, 1, 2, 3}

### **3.2 Problem Format**

```python
{
    "id": "abs_val_001",
    "category": "absolute_value",
    "problem": "Solve the equation |x - 3| = 5",
    "required_cases": [
        "x - 3 = 5 (when x - 3 >= 0)",
        "-(x - 3) = 5 (when x - 3 < 0)"
    ],
    "ground_truth_solution": ["x = 8", "x = -2"],
    "explanation": "Split into cases based on sign of (x-3)"
}
```

### **3.3 Dataset Creation**

**Phase 1**: Create 50 problems manually
- 15 absolute value
- 15 piecewise functions
- 10 sign analysis
- 10 range-based

**Phase 2** (if needed): Expand to 100 problems

**Storage**: JSON file `data/math_case_analysis.json`

---

## 4. Prompt Templates

### **4.1 Baseline: Standard Chain-of-Thought**

```
Problem: {problem}

Let's solve this step by step:
```

**Characteristics**:
- Simple instruction
- No explicit MECE guidance
- Natural reasoning flow

### **4.2 MECE Prompting**

**Version 1: Explicit MECE Instruction**
```
Problem: {problem}

Solve this problem using the MECE principle (Mutually Exclusive, Collectively Exhaustive):

1. Identify all distinct cases that need to be considered
2. Ensure cases don't overlap (mutually exclusive)
3. Ensure all possibilities are covered (collectively exhaustive)
4. Solve each case separately
5. Combine results

Let's solve this step by step:
```

**Version 2: Case Analysis Template**
```
Problem: {problem}

Analyze this problem by breaking it into cases:

Step 1: Identify what cases need to be considered
Step 2: For each case, state the condition clearly
Step 3: Verify cases don't overlap
Step 4: Verify all possibilities are covered
Step 5: Solve each case
Step 6: Combine solutions

Let's work through this:
```

**Version 3: Structured MECE** (if versions 1-2 underperform)
```
Problem: {problem}

Use structured case analysis:

CASE IDENTIFICATION:
- List all cases needed: [...]

MUTUAL EXCLUSIVITY CHECK:
- Verify no cases overlap: [...]

EXHAUSTIVENESS CHECK:
- Verify all possibilities covered: [...]

CASE SOLUTIONS:
- Case 1: [condition] → [solution]
- Case 2: [condition] → [solution]
...

FINAL ANSWER:
```

### **4.3 Prompt Selection Strategy**

1. Start with Version 1 (explicit MECE)
2. If 0.6B model struggles with structure, try Version 2 (template)
3. If needed, use Version 3 (highly structured)
4. Pilot test with 5 problems to select best prompt

---

## 5. Implementation Architecture

### **5.1 Directory Structure**

```
mece_step_by_step/
├── CLAUDE.md
├── RESEARCH.md
├── REVIEW_AND_CORRECTIONS.md
├── IMPLEMENTATION.md (this file)
├── data/
│   ├── math_case_analysis.json          # Problem dataset
│   └── results/                          # Evaluation results
│       ├── baseline_results.json
│       └── mece_results.json
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── qwen_inference.py            # Qwen3-0.6B inference (MLX/llama.cpp)
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── baseline_prompt.py
│   │   └── mece_prompt.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── mutual_exclusivity.py
│   │   ├── collective_exhaustiveness.py
│   │   └── accuracy.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   └── parser.py                    # Parse reasoning steps
│   └── utils/
│       ├── __init__.py
│       └── config.py
├── scripts/
│   ├── create_dataset.py                # Generate problem dataset
│   ├── run_evaluation.py                # Main evaluation script
│   └── analyze_results.py               # Result analysis & visualization
├── notebooks/
│   └── exploratory_analysis.ipynb       # Interactive analysis
├── tests/
│   ├── test_metrics.py
│   └── test_parsers.py
├── pyproject.toml                        # uv package config
├── requirements.txt                      # Fallback for pip
└── README.md
```

### **5.2 Core Components**

#### **Model Inference** (`src/models/qwen_inference.py`)

```python
from typing import List, Dict
import mlx.core as mx  # or llama_cpp

class QwenInference:
    """Qwen3-0.6B inference optimized for M4 Max"""

    def __init__(self, model_path: str = "Qwen/Qwen3-0.6B"):
        # Load model with MLX optimization for Apple Silicon
        self.model = self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load Qwen3-0.6B with MLX or llama.cpp"""
        # MLX: Optimized for Apple Silicon
        # Alternatively: Use transformers with MPS backend
        pass

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate reasoning response"""
        pass
```

#### **MECE Metrics** (`src/metrics/`)

Implement the metric functions defined in Section 2.

#### **Evaluator** (`src/evaluation/evaluator.py`)

```python
class MECEEvaluator:
    """Main evaluation harness"""

    def __init__(self, model: QwenInference, dataset: List[Dict]):
        self.model = model
        self.dataset = dataset

    def evaluate(
        self,
        prompt_type: str = "baseline"  # or "mece"
    ) -> Dict[str, Any]:
        """
        Run evaluation on dataset

        Returns:
            {
                'accuracy': float,
                'avg_me_score': float,
                'avg_ce_score': float,
                'avg_mece_score': float,
                'detailed_results': List[Dict]
            }
        """
        results = []
        for problem in tqdm(self.dataset):
            result = self.evaluate_single(problem, prompt_type)
            results.append(result)

        return self.aggregate_results(results)

    def evaluate_single(
        self,
        problem: Dict,
        prompt_type: str
    ) -> Dict:
        """Evaluate single problem"""
        # Generate response
        prompt = create_prompt(problem, prompt_type)
        response = self.model.generate(prompt)

        # Parse reasoning steps
        steps = parse_reasoning_steps(response)

        # Compute metrics
        me_metrics = compute_mutual_exclusivity(steps)
        ce_metrics = compute_collective_exhaustiveness(steps, problem)
        accuracy = compute_accuracy(response, problem['ground_truth_solution'])

        return {
            'problem_id': problem['id'],
            'response': response,
            'steps': steps,
            'me_metrics': me_metrics,
            'ce_metrics': ce_metrics,
            'accuracy': accuracy,
            'mece_score': compute_mece_score(
                me_metrics['embedding_me_score'],
                ce_metrics['ce_score']
            )
        }
```

---

## 6. Implementation Phases

### **Phase 1: Setup & Dataset Creation** (Day 1)

**Tasks**:
1. ✅ Set up project structure
2. Create `pyproject.toml` for uv
3. Install dependencies:
   - `mlx` or `llama-cpp-python` (model inference)
   - `transformers` (model loading)
   - `sentence-transformers` (embeddings)
   - `torch` (if needed)
   - `numpy`, `pandas`, `matplotlib`, `seaborn`
   - `pytest` (testing)
4. Create 50 math case analysis problems in `data/math_case_analysis.json`
5. Validate problem format and ground truth

**Deliverable**: Dataset with 50 problems, validated and ready

### **Phase 2: Model Setup** (Day 1-2)

**Tasks**:
1. Download Qwen3-0.6B from HuggingFace
2. Test inference on M4 Max (MLX preferred for speed)
3. Benchmark inference speed (should be fast!)
4. Implement `QwenInference` class
5. Test with sample prompts

**Deliverable**: Working Qwen3-0.6B inference on M4 Max

### **Phase 3: Prompt Engineering** (Day 2)

**Tasks**:
1. Implement baseline CoT prompt
2. Implement MECE prompt (Version 1)
3. Test both on 5 sample problems
4. Analyze outputs qualitatively
5. Adjust prompts if needed

**Deliverable**: Finalized prompt templates

### **Phase 4: Metric Implementation** (Day 2-3)

**Tasks**:
1. Implement mutual exclusivity metrics
   - Embedding-based similarity
   - Case condition overlap parser
2. Implement collective exhaustiveness metrics
   - Case enumeration
   - Condition coverage checker
3. Implement accuracy metric
4. Write unit tests for all metrics
5. Validate on sample problems

**Deliverable**: Complete metric calculation pipeline, tested

### **Phase 5: Evaluation Pipeline** (Day 3)

**Tasks**:
1. Implement `MECEEvaluator` class
2. Implement response parser (extract reasoning steps)
3. Create evaluation scripts
4. Test end-to-end pipeline on 5 problems
5. Debug and fix issues

**Deliverable**: Working end-to-end evaluation pipeline

### **Phase 6: Full Evaluation** (Day 3-4)

**Tasks**:
1. Run baseline evaluation (50 problems)
2. Run MECE evaluation (50 problems)
3. Save results to JSON
4. Initial results inspection

**Deliverable**: Complete evaluation results for both conditions

### **Phase 7: Analysis & Visualization** (Day 4-5)

**Tasks**:
1. Implement analysis scripts
2. Generate comparison tables
3. Create visualizations:
   - Accuracy comparison (bar chart)
   - ME/CE score distributions (box plots)
   - Per-category breakdowns
   - Case coverage heatmap
4. Statistical significance tests (t-test, Mann-Whitney U)
5. Qualitative analysis of failure cases

**Deliverable**: Complete analysis with visualizations

### **Phase 8: Documentation & Writeup** (Day 5)

**Tasks**:
1. Write RESULTS.md with findings
2. Update README.md
3. Document code
4. Create example notebook
5. Prepare summary for next steps

**Deliverable**: Documented results and conclusions

---

## 7. Success Criteria

### **Minimum Viable Success**:
1. ✅ Pipeline runs end-to-end without errors
2. ✅ Metrics are computable for all 50 problems
3. ✅ Both conditions (baseline and MECE) evaluated
4. ✅ Results show measurable differences in ME/CE scores

### **Strong Success**:
1. ✅ MECE prompting shows higher ME score (>10% improvement)
2. ✅ MECE prompting shows higher CE score (>10% improvement)
3. ✅ MECE prompting shows higher or equal accuracy
4. ✅ Results are statistically significant (p < 0.05)

### **Outstanding Success**:
1. ✅ All strong success criteria met
2. ✅ MECE prompting improves accuracy by >5%
3. ✅ Clear patterns in when MECE helps vs doesn't help
4. ✅ Generalizable insights for other domains

---

## 8. Experimental Design Details

### **8.1 Comparison Setup**

**Between-Problems Design**:
- Same 50 problems for both conditions
- Randomize evaluation order to avoid order effects
- Fixed random seed for reproducibility

**Variables**:
- **Independent Variable**: Prompt type (baseline vs MECE)
- **Dependent Variables**:
  - Answer accuracy (primary)
  - ME score (secondary)
  - CE score (secondary)
  - MECE score (combined)

**Controls**:
- Same model (Qwen3-0.6B)
- Same temperature (0.7)
- Same max_tokens (512)
- Same random seed

### **8.2 Analysis Plan**

**Quantitative Analysis**:
1. Descriptive statistics (mean, std, median)
2. Comparison tests:
   - Paired t-test for accuracy
   - Wilcoxon signed-rank test (non-parametric)
3. Effect size (Cohen's d)
4. Per-category breakdown

**Qualitative Analysis**:
1. Manual inspection of 10 baseline responses
2. Manual inspection of 10 MECE responses
3. Categorize common errors
4. Identify patterns in MECE benefits

---

## 9. Expected Challenges & Mitigation

### **Challenge 1: Small Model Limitations**
- **Issue**: Qwen3-0.6B may struggle with complex instructions
- **Mitigation**:
  - Start with simple MECE prompt (Version 1)
  - If needed, try more structured Version 2-3
  - Focus on simpler problem categories first

### **Challenge 2: Response Parsing**
- **Issue**: Extracting reasoning steps may be error-prone
- **Mitigation**:
  - Design robust parsers with fallbacks
  - Manual validation on subset
  - Log unparseable responses for debugging

### **Challenge 3: Case Extraction**
- **Issue**: Identifying cases in free-form text is hard
- **Mitigation**:
  - Use pattern matching (regex)
  - Look for keywords: "case", "when", "if", "for"
  - Embedding-based similarity as fallback

### **Challenge 4: Metric Validation**
- **Issue**: Are our metrics measuring the right thing?
- **Mitigation**:
  - Human annotation on 20 problems
  - Compare automated metrics to human judgment
  - Adjust thresholds if needed

### **Challenge 5: M4 Max Performance**
- **Issue**: Inference might be slower than expected
- **Mitigation**:
  - Use MLX for Apple Silicon optimization
  - Batch processing if needed
  - 0.6B should be fast enough (~5-10 sec/problem)

---

## 10. Configuration Management

### **config.yaml**

```yaml
model:
  name: "Qwen/Qwen3-0.6B"
  framework: "mlx"  # or "llama.cpp" or "transformers"
  device: "mps"  # Metal Performance Shaders for M4
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9

dataset:
  path: "data/math_case_analysis.json"
  categories:
    - absolute_value
    - piecewise_functions
    - sign_analysis
    - range_based

evaluation:
  prompt_types:
    - baseline
    - mece
  random_seed: 42
  batch_size: 1  # Sequential for now

metrics:
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  similarity_threshold: 0.8  # For overlap detection

output:
  results_dir: "data/results"
  save_responses: true
  save_metrics: true
```

---

## 11. Next Steps After Initial Results

### **If MECE Shows Promise**:
1. Expand to 100 problems
2. Test with larger model (Qwen3-1.7B or Qwen3-4B)
3. Add second domain (logic puzzles)
4. Try coverage oracle for CE measurement

### **If Results are Mixed**:
1. Analyze which problem types benefit
2. Refine prompts for specific categories
3. Investigate failure modes
4. Compare to human-written MECE decompositions

### **If MECE Doesn't Help**:
1. Analyze why (model too small? prompts unclear?)
2. Test with larger model to isolate model size effect
3. Consider alternative approaches (fine-tuning, few-shot)
4. Document findings as negative result (still valuable!)

---

## 12. Timeline Estimate

**Total**: ~5-7 days of focused work

| Phase | Duration | Key Deliverable |
|-------|----------|----------------|
| 1. Setup & Dataset | 1 day | 50 problems |
| 2. Model Setup | 0.5 day | Working inference |
| 3. Prompt Engineering | 0.5 day | Final prompts |
| 4. Metric Implementation | 1 day | Tested metrics |
| 5. Evaluation Pipeline | 0.5 day | End-to-end pipeline |
| 6. Full Evaluation | 0.5 day | Both condition results |
| 7. Analysis | 1 day | Visualizations & stats |
| 8. Documentation | 0.5 day | RESULTS.md |
| **Buffer** | 0.5 day | Debugging |

**Note**: With M4 Max and 0.6B model, inference should be very fast, so most time is setup and analysis.

---

## 13. Reproducibility Checklist

- [ ] All random seeds fixed (model, data shuffling, etc.)
- [ ] Model version pinned (Qwen3-0.6B specific commit/version)
- [ ] Dependencies pinned in `pyproject.toml`
- [ ] Configuration saved with results
- [ ] All prompts logged
- [ ] Raw model outputs saved
- [ ] Code version controlled (git)
- [ ] README includes exact reproduction steps
- [ ] Hardware specs documented (M4 Max, RAM, etc.)

---

## 14. Optional Extensions

### **If Time Permits**:

1. **Few-Shot MECE Examples**
   - Add 2-3 MECE examples to prompt
   - Compare zero-shot vs few-shot MECE

2. **Contrastive Analysis**
   - For failed problems, show what MECE structure would be
   - Analyze gap between model output and ideal MECE

3. **Interactive Demo**
   - Streamlit app for interactive MECE prompting
   - Side-by-side comparison of baseline vs MECE

4. **Ablation Studies**
   - Test each MECE component separately (ME only, CE only)
   - Identify which aspect matters more

---

## Summary

This implementation plan is:
- ✅ **Hardware-appropriate**: Optimized for M4 Max with 0.6B model
- ✅ **Focused**: Single domain (math), clear comparison (MECE vs baseline)
- ✅ **Tractable**: ~5-7 days, 50 problems, straightforward metrics
- ✅ **Rigorous**: Computational metrics, statistical testing, reproducible
- ✅ **Flexible**: Multiple prompt versions, clear success criteria
- ✅ **Extensible**: Clear next steps based on results

**Ready to proceed with Phase 1: Setup & Dataset Creation!**
