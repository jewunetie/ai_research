# Pivot 9: World Models & Digital Simulation

**Research Date**: 2025-11-22
**Viability**: ⭐⭐⭐ (Medium - Ambitious, Long-Term)

---

## Executive Summary

Train byte-level models to **simulate digital environments** by predicting future file states from byte sequences. Inspired by bGPT's finding that byte models can simulate CPU states, this pivot explores whether models can learn the "physics" of digital systems (file systems, programs, network protocols) by modeling byte-level state transitions.

**Vision**: A model that predicts "what happens next" when you edit a file, run a program, or send network packets - all at the byte level.

---

## 1. Concept: Digital World Simulation

### Traditional World Models (Robotics/RL)
```
Observation (pixels) → Model → Predict next observation (pixels)
                ↑
         Physical world simulation (gravity, friction, collisions)
```

### Digital World Models (Byte-Level)
```
File state (bytes) → Model → Predict next file state (bytes)
         ↑
  Learn rules of computation (file operations, program execution, network)
```

**Examples**:
- **File Edit**: Given `file.txt` bytes + edit command → Predict modified `file.txt` bytes
- **Program Execution**: Given `program.py` bytes + input → Predict output bytes
- **Network Protocol**: Given HTTP request bytes → Predict HTTP response bytes
- **Git Operations**: Given repo bytes + `git commit` → Predict updated repo bytes

---

## 2. Precedent: bGPT CPU State Modeling

**bGPT (2024)** demonstrated byte models can simulate **CPU states**:
- Input: CPU registers and memory (as bytes)
- Task: Predict next CPU state after executing instruction
- **Result**: bGPT learned to simulate x86 instruction execution

**Implication**: Byte models can learn computational rules from data alone

---

## 3. Potential Architectures

### 3.1 File System Simulator
```python
class FileSystemByteModel(nn.Module):
    def __init__(self):
        self.state_encoder = ByteTransformer(max_bytes=65536)  # Current file system state
        self.action_encoder = nn.Embedding(num_actions=100, embed_dim=256)  # Operations (read, write, delete)
        self.predictor = TransformerDecoder(embed_dim=768, depth=12)

    def forward(self, fs_state_bytes, action):
        """
        fs_state_bytes: [batch, seq_len] - Current directory listing + file contents
        action: [batch] - File operation (e.g., 'write 100 bytes to file.txt')
        Returns: [batch, seq_len, 256] - Predicted next file system state
        """
        state_features = self.state_encoder(fs_state_bytes)
        action_features = self.action_encoder(action)
        next_state_logits = self.predictor(state_features, action_features)
        return next_state_logits
```

### 3.2 Program Execution Predictor
```python
class ProgramByteModel(nn.Module):
    def forward(self, program_bytes, input_bytes):
        """
        program_bytes: Python script as bytes
        input_bytes: stdin as bytes
        Returns: Predicted stdout bytes
        """
        program_embedding = self.encoder(program_bytes)
        input_embedding = self.encoder(input_bytes)
        output_prediction = self.decoder(program_embedding, input_embedding)
        return output_prediction  # Predicted program output
```

**Challenge**: Would need to learn Python semantics, libraries, etc. from byte patterns alone (extremely difficult!)

---

## 4. Feasible Experiments

### 4.1 Simple File Operations (Most Feasible)
**Setup**:
- Simulate file system with 10-100 files
- Operations: create, delete, append, rename
- Train model to predict file system state after operation

**Dataset Generation**:
```python
# Generate 100K training examples
for i in range(100000):
    fs_state = random_file_system()  # Random files and contents
    operation = random_operation()   # Random file operation
    next_state = simulate(fs_state, operation)  # Ground truth
    dataset.append((fs_state, operation, next_state))
```

**Expected**: Model learns basic file system rules (files persist after creation, delete removes files, etc.)

### 4.2 Network Protocol Simulation (Medium Feasibility)
**Setup**:
- HTTP request/response pairs
- Model predicts response bytes given request bytes
- Test: Does model learn HTTP headers, status codes, content-length?

**Dataset**: HTTP logs from web servers (Apache, nginx)

### 4.3 Git Repository Evolution (Ambitious)
**Setup**:
- Input: Git repo state (all files as bytes) + commit diff
- Output: Predict updated repo state
- Test: Can model "understand" version control?

---

## 5. Use Cases

1. **Automated Testing**: Predict program output without execution (faster CI/CD)
2. **File Recovery**: Predict file state before corruption (data recovery)
3. **Network Debugging**: Simulate protocol behavior for debugging
4. **Security**: Predict attack outcomes (exploit analysis)
5. **Code Generation**: Generate byte-level program outputs (complement to code LLMs)

---

## 6. Challenges

1. **Computational Complexity**: Simulating programs at byte level is intractable
2. **Ambiguity**: Many operations have complex, context-dependent outcomes
3. **Long Sequences**: File systems can be gigabytes (cannot fit in memory)
4. **Evaluation**: Hard to measure "simulation quality"

---

## 7. Realistic Scope

**Start Simple**:
- File system simulation (create/delete/append only)
- Fixed-size files (max 1KB each)
- Deterministic operations (no randomness)

**Expand If Successful**:
- Network protocol simulation (HTTP)
- Simple program execution (calculator, string operations)

**Skip**:
- Full program execution (too complex)
- Real-world file systems (too large)

---

## 8. Success Criteria

- [ ] Predict file system state after operations (≥90% byte-level accuracy)
- [ ] Learn file persistence (files exist across operations)
- [ ] Generalize to unseen file names/contents
- [ ] Demonstrate emergent understanding of file system rules

---

## 9. Viability: ⭐⭐⭐ (Medium)

**Strengths**:
- bGPT proves byte models can learn computational rules
- Clear evaluation (predict-then-verify)
- Novel research direction

**Challenges**:
- Computational cost of simulating complex systems
- Limited practical value (easier to just run the actual operation)
- Difficult to scale beyond toy examples

**Recommendation**: Pursue as **research-focused project** if interested in emergent computational understanding. Limited immediate practical value, but high scientific interest.

**Timeline**: 4-6 weeks (file system simulation), 8-10 weeks (network protocol)

---

## References

1. bGPT: "Beyond Language Models: Byte Models are Digital World Simulators", Feb 2024
2. World Models (Ha & Schmidhuber): Learning forward models for RL
3. DreamerV3: World models for reinforcement learning
4. Transformers as Program Synthesizers (e.g., AlphaCode)
