# Pivot 8: File Format Synthesis & Fuzzing

**Research Date**: 2025-11-22
**Viability**: ⭐⭐⭐ (Medium - Security & Testing Applications)

---

## Executive Summary

Use byte-level generative models to **synthesize valid image/audio/document files** and generate test cases for **fuzzing file parsers**. Unlike traditional fuzzers (random bit flips), byte-level models learn file format structure and generate semantically valid but unusual files that expose parser bugs.

**Key Application**: Automated security testing of image decoders, audio codecs, PDF readers.

---

## 1. Concept

### Traditional Fuzzing
```
Valid PNG → Random bit flips → Invalid PNG → Feed to parser → Crash?
              ↑
        Mostly garbage, low hit rate
```

### Learned Byte-Level Fuzzing
```
Corpus of PNGs → Byte-level VAE → Sample latent → Generate novel PNG → Parser test
                       ↑
                 Generates mostly valid PNGs with subtle variations
                 (Header intact, chunk structure valid, data slightly unusual)
```

---

## 2. Architecture

### Byte-Level Variational Autoencoder
```python
class ByteFileVAE(nn.Module):
    def __init__(self, max_bytes=8192, latent_dim=256):
        # Encoder: File bytes → latent distribution
        self.encoder = ByteTransformer(max_bytes, embed_dim=512)
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)

        # Decoder: Latent → file bytes
        self.decoder = ByteTransformerDecoder(latent_dim, max_bytes)
        self.byte_head = nn.Linear(512, 256)  # Predict byte values

    def forward(self, file_bytes):
        # Encode
        features = self.encoder(file_bytes).mean(dim=1)
        mu, logvar = self.mu_head(features), self.logvar_head(features)

        # Reparameterization
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        # Decode
        reconstructed_logits = self.decoder(z)  # [batch, max_bytes, 256]
        return reconstructed_logits, mu, logvar

    def generate(self, num_samples=10):
        """Generate novel file bytes"""
        z = torch.randn(num_samples, latent_dim)
        generated_logits = self.decoder(z)
        generated_bytes = generated_logits.argmax(dim=-1)  # [num_samples, max_bytes]
        return generated_bytes
```

---

## 3. Use Cases

### 3.1 Fuzzing Image Decoders
**Target**: libpng, libjpeg, WebP decoder
**Approach**:
1. Train byte-VAE on 10K valid PNG/JPEG files
2. Generate 100K synthetic files by sampling latent space
3. Feed to decoder, catch crashes/memory leaks
4. Prioritize files that cause unusual execution paths

**Expected**: Find edge cases missed by random fuzzing (e.g., valid headers but unusual chunk ordering)

### 3.2 Format-Aware Mutation
**Concept**: Learn format structure, then mutate intelligently
- Preserve PNG header (`\x89PNG\r\n\x1a\n`)
- Mutate chunk data while keeping CRC valid
- Generate valid-but-unusual color palettes, compression levels

### 3.3 Adversarial File Generation
**Goal**: Generate files that maximize decoder processing time (DoS attacks)
- Train VAE with reward for high decoding latency
- Discover pathological cases (e.g., PNG with excessive filters)

---

## 4. Implementation Roadmap

**Week 1**: Collect dataset
- Scrape 10K PNG/JPEG files from ImageNet, COCO
- Include diverse formats (grayscale, RGB, RGBA, progressive JPEG)

**Week 2**: Train byte-VAE
- Implement architecture above
- Train with reconstruction loss + KL divergence
- Validate: Do generated files decode correctly?

**Week 3**: Fuzzing integration
- Integrate with AFL (American Fuzzy Lop) or LibFuzzer
- Generate seed corpus from VAE
- Run fuzzing campaign on libpng/libjpeg

**Week 4**: Analysis
- Count crashes, unique bugs found
- Compare vs random fuzzing, traditional mutation-based fuzzing
- Analyze what makes learned fuzzing effective

---

## 5. Advantages

1. **Higher Hit Rate**: Generated files more likely to be valid (trigger deeper code paths)
2. **Format-Aware**: Preserves critical structures (headers, checksums)
3. **Automated**: No manual format specification required
4. **Transferable**: Train once, fuzz many parsers

---

## 6. Challenges

1. **Format Complexity**: VAE may not perfectly learn all format constraints
2. **Checksum/CRC**: Models must learn to generate valid checksums (hard!)
3. **Evaluation**: Hard to measure "quality" of generated test cases
4. **Comparison**: Traditional fuzzers are highly optimized (high bar to beat)

---

## 7. Hybrid Approach (Practical)

```
1. Use byte-VAE to generate "plausible" files
2. Apply light mutation (bit flips, byte swaps) to VAE outputs
3. Use as seed corpus for traditional coverage-guided fuzzer (AFL++)
```

**Advantage**: Combines learned structure with traditional coverage feedback

---

## 8. Success Criteria

- [ ] VAE generates ≥80% valid PNG/JPEG files
- [ ] Find ≥1 unique bug in libpng or libjpeg
- [ ] Demonstrate higher code coverage than random fuzzing
- [ ] Shorter time-to-first-bug vs random fuzzing

---

## 9. Related Work

- **DeepFuzz**: Neural fuzzing for image parsers (exists, baseline for comparison)
- **Nautilus**: Grammar-based fuzzing (rule-based, not learned)
- **AFL++**: State-of-the-art coverage-guided fuzzer
- **OSS-Fuzz**: Google's continuous fuzzing service

---

## 10. Viability: ⭐⭐⭐ (Medium)

**Strengths**:
- Clear application (security testing)
- Existing fuzzing infrastructure to build on
- Byte-level VAE is tractable (simpler than GANs)

**Challenges**:
- Competing with highly optimized traditional fuzzers
- Difficult to learn perfect format constraints
- Validation: Does learned fuzzing actually find more bugs?

**Recommendation**: Pursue as side project or collaboration with security researchers. Medium research contribution, high practical value if successful.

**Timeline**: 3-4 weeks (proof-of-concept), 6-8 weeks (full evaluation)

---

## References

1. "DeepFuzz: Automatic Generation of Syntax Valid C Programs for Fuzz Testing", AAAI 2019
2. American Fuzzy Lop (AFL): http://lcamtuf.coredump.cx/afl/
3. LibFuzzer: https://llvm.org/docs/LibFuzzer.html
4. OSS-Fuzz: https://google.github.io/oss-fuzz/
