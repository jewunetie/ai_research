# Bug Report - Phase 1 Implementation

## Critical Bugs Found

### Bug #1: Calculator `b` parameter incorrectly optional
**File**: `src/tcoml/agent/tool_registry.py`
**Lines**: 30, 59

**Issue**:
- Line 30: `def calculator(operation: str, a: float, b: float = None)`
- Line 59: `ToolParameter(name="b", ... required=False)`
- But ALL operations (add, subtract, multiply, divide, percentage) use `b`
- If `b` is None, will get TypeError: `unsupported operand type(s) for +: 'float' and 'NoneType'`

**Fix**: Make `b` required and remove default value

---

### Bug #2: Mock LLM checks "calculate" before "percentage"
**File**: `src/tcoml/llm/mock_client.py`
**Lines**: 76-86 vs 102-112

**Issue**:
- Line 76-86: Checks for "calculate" keyword → returns "add" operation
- Line 102-112: Checks for percentage keywords
- For "Calculate 10% tip on $131.49", matches "calculate" first, returns wrong operation!
- **This caused the test failure**: Expected 13.15, got 141.49 (10+131.49)

**Fix**: Check for percentage BEFORE generic "calculate" keyword

---

### Bug #3: Percentage argument order logic is backwards
**File**: `src/tcoml/llm/mock_client.py`
**Lines**: 109-110

**Issue**:
```python
"a": float(numbers[1] if "%" in user_message[:user_message.find(numbers[0])] else numbers[0]),
"b": float(numbers[0] if "%" in user_message[:user_message.find(numbers[0])] else numbers[1])
```

For "Calculate 10% tip on $131.49":
- numbers = ['10', '131.49']
- "%" is NOT before '10', so: a=10, b=131.49
- But calculator expects: a=base, b=percentage
- Should be: a=131.49, b=10

The logic is backwards!

**Better fix**: Use regex to find which number has "%" after it

---

## All Bugs Fixed ✅

### Bug #1 Fix: tool_registry.py
```python
# BEFORE
def calculator(operation: str, a: float, b: float = None) -> float:
ToolParameter(name="b", ... required=False)

# AFTER
def calculator(operation: str, a: float, b: float) -> float:
ToolParameter(name="b", ... required=True)
```

### Bug #2 & #3 Fix: mock_client.py
Reordered checks and fixed percentage logic:
```python
# Check percentage FIRST (before "calculate")
if "%" in user_message or "percent" in user_lower or "tip" in user_lower:
    # Use regex to find which number has % after it
    percentage_match = re.search(r'(\d+(?:\.\d+)?)%', user_message)
    if percentage_match:
        percentage_value = float(percentage_match.group(1))
        base_value = float(numbers[1]) if float(numbers[0]) == percentage_value else float(numbers[0])
    # Correctly assign: a=base, b=percentage
```

## Test Results After Fixes

```
[Test 3] Testing Full Workflow...
  Task: Calculate 10% tip on $131.49
  Expected: 13.15
  LLM generated 1 tool calls
  Executed: calculator({'operation': 'percentage', 'a': 131.49, 'b': 10.0}) → 13.15
✓ Task completed. Success: True  ← FIXED! Was False before
```

**Before Fix**: Mock LLM used 'add' operation (141.49 = 10+131.49) ❌
**After Fix**: Mock LLM uses 'percentage' operation (13.15 = 131.49 * 10%) ✅

All 21 tests passing: ✅ 100% success rate
