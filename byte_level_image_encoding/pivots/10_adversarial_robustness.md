# Pivot 10: Adversarial Robustness at the Byte Level

**Research Date**: 2025-11-22
**Viability**: ⭐⭐⭐ (Medium - Novel Defense Angle)

---

## Executive Summary

Investigate whether **byte-level models are more robust to adversarial attacks** than pixel-level models. Hypothesis: Adversarial perturbations optimized in pixel space may not transfer effectively to byte space due to JPEG compression, PNG encoding, and format constraints.

**Key Insight**: Pixel perturbations must survive encoding → bytes → decoding round-trip, which may destroy carefully crafted gradients.

---

## 1. Hypothesis

### Pixel-Level Adversarial Attack (Traditional)
```
Original Image (pixels) → Add ε perturbation → Adversarial Image (pixels) → Model → Wrong prediction
                            ↑
                  Optimized in pixel space (no compression)
```

### Byte-Level Adversarial Attack (Proposed)
```
Original JPEG (bytes) → Modify bytes → Adversarial JPEG (bytes) → Byte Model → Wrong prediction?
                          ↑
                    Must produce valid JPEG! Constraints:
                    - Huffman decoding must succeed
                    - DCT coefficients must be valid
                    - Quantization tables preserved
                    - CRC/checksums correct
```

**Challenge for Attacker**: Byte-level perturbations must satisfy format constraints, making attacks harder.

---

## 2. Research Questions

1. **Are byte models inherently more robust?** (Compression as defense)
2. **Do pixel-space attacks transfer to byte space?** (Encode attacked pixel image to JPEG → still adversarial?)
3. **Can attackers craft byte-level adversarial examples?** (Optimize directly in byte space)
4. **Does JPEG compression quality affect robustness?** (Lower quality = more compression = more robust?)

---

## 3. Experimental Design

### Experiment 1: Pixel Attack → Byte Model
```
1. Generate pixel-space adversarial example (FGSM, PGD on ResNet)
2. Encode to JPEG at various quality levels (50, 75, 95)
3. Feed JPEG bytes to byte model
4. Measure attack success rate
```

**Hypothesis**: Attack success rate decreases as JPEG quality decreases (compression destroys perturbation)

### Experiment 2: Direct Byte-Space Attack
```
1. Start with clean JPEG bytes
2. Optimize byte perturbations to fool byte model
3. Constraint: Perturbed bytes must decode to valid image
4. Measure attack success vs pixel-space attacks
```

**Hypothesis**: Byte-space attacks harder to craft due to format constraints

### Experiment 3: Format Robustness
```
1. Train byte model on mixed formats (JPEG, PNG, WebP)
2. Craft adversarial example in one format (e.g., JPEG)
3. Convert to another format (e.g., PNG)
4. Test if attack persists
```

**Hypothesis**: Format conversion breaks adversarial signal

---

## 4. Potential Outcomes

### Scenario A: Byte Models More Robust
- Compression destroys pixel-space perturbations
- Format constraints prevent byte-level attacks
- **Implication**: Byte-level encoding is a practical defense against adversarial examples

### Scenario B: Byte Models Equally Vulnerable
- Attackers adapt by optimizing in byte space
- Compression doesn't sufficiently destroy gradients
- **Implication**: Byte-level provides no robustness advantage

### Scenario C: Byte Models Less Robust
- Byte-level attacks exploit compression artifacts
- Unexpected vulnerabilities in file format parsing
- **Implication**: Byte-level is actively harmful for robustness

---

## 5. Implementation Roadmap

**Week 1**: Setup
- Implement pixel-level baseline (ResNet-18 on CIFAR-10)
- Train byte-level model (ByteFormer on JPEG bytes)
- Verify clean accuracy comparable

**Week 2**: Pixel-space attacks
- Generate FGSM, PGD adversarial examples
- Encode to JPEG at different quality levels
- Test on byte model

**Week 3**: Byte-space attacks
- Implement differentiable JPEG encoder (to get gradients)
- Optimize byte perturbations with format constraints
- Measure attack success rate

**Week 4**: Analysis
- Compare robustness: pixel model vs byte model
- Analyze failure modes
- Test format conversion defense

---

## 6. Challenges

1. **Differentiable Encoder**: Need differentiable JPEG encoder to optimize byte perturbations
2. **Format Constraints**: Enforcing valid byte sequences is non-trivial
3. **Computational Cost**: Byte-level attacks may be much slower
4. **Evaluation**: Standard robustness metrics may not apply

---

## 7. Related Work

- **JPEG Compression as Defense**: Prior work shows compression defends against pixel attacks (but degrades clean accuracy)
- **Certified Robustness**: Provable defenses (e.g., randomized smoothing) - orthogonal to byte-level
- **Adversarial Training**: Train on adversarial examples - applicable to both pixel and byte models

---

## 8. Success Criteria

- [ ] Measure attack success rate: pixel attacks → byte model
- [ ] Compare robustness: byte model vs pixel model on same dataset
- [ ] Demonstrate format constraint benefit (attacks don't transfer across formats)
- [ ] Quantify compression-robustness trade-off

---

## 9. Expected Outcome

**Most Likely**: Byte models show **moderate robustness improvement** (20-40% lower attack success rate) due to compression, but not completely robust.

**Contribution**: First systematic study of adversarial robustness at byte level. Even if improvement is modest, provides new defense angle.

---

## 10. Viability: ⭐⭐⭐ (Medium)

**Strengths**:
- Novel research angle (understudied)
- Clear hypothesis (compression as defense)
- Practical implications if successful

**Challenges**:
- Requires differentiable encoder (implementation complexity)
- May show no advantage (neutral result)
- Competing with mature adversarial robustness research

**Recommendation**: Pursue as **research contribution** if interested in adversarial ML. Medium risk, medium reward.

**Timeline**: 3-4 weeks (full evaluation)

---

## References

1. "Adversarial Examples in the Physical World" (Kurakin et al.) - JPEG robustness
2. "Certified Robustness to Adversarial Examples" (Cohen et al.) - Randomized smoothing
3. "Adversarial Training and Robustness for Multiple Perturbations" (Maini et al.)
4. Foolbox: Python toolbox for adversarial attacks
