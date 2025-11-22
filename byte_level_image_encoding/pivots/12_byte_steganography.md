# Pivot 12: Byte-Level Steganography

**Research Date**: 2025-11-22
**Viability**: ⭐⭐ (Low-Medium - Niche Application)

---

## Executive Summary

Use byte-level models to **hide and detect secret messages in image files** at the byte level (not pixel level). Unlike traditional steganography (modify least significant bits of pixels), byte-level steganography embeds data in file format structures (metadata, compression tables, padding bytes).

**Key Idea**: Exploit JPEG/PNG format redundancy to hide data while preserving visual appearance and file validity.

---

## 1. Byte vs Pixel Steganography

### Traditional LSB Steganography (Pixel-Level)
```
Image pixels → Modify least significant bits → Save as PNG → Extract LSBs to recover message
                     ↑
              Detectable via statistical analysis (histogram anomalies)
```

### Byte-Level Steganography (Format-Level)
```
JPEG file bytes → Modify Huffman tables, quantization tables, padding, or comments
                       ↑
                 Embed message in format metadata, harder to detect
```

**Advantages**:
- **Format-Native**: Message hidden in file structure, not pixels
- **Robustness**: Survives pixel-level transformations (resize, crop)
- **Detectability**: Harder to detect than LSB (no pixel statistics changed)

---

## 2. Hiding Locations in JPEG

### JPEG File Structure
```
[SOI marker] [APP0 JFIF] [DQT quantization] [SOF frame info]
[DHT Huffman tables] [SOS scan data] ... [EOI marker]
```

**Embedding Opportunities**:
1. **Comment Segments (COM)**: Arbitrary data allowed (obvious, easy to detect)
2. **Quantization Tables (DQT)**: Slight modifications imperceptible
3. **Huffman Table Ordering**: Reorder entries (same decoding, different bytes)
4. **Padding Bytes**: Some JPEG decoders ignore extra bytes in segments
5. **DCT Coefficients**: Modify high-frequency coefficients (subtle visual change)

---

## 3. Neural Byte-Level Steganography

### 3.1 Encoder: Hide Message
```python
class ByteStegoEncoder(nn.Module):
    def __init__(self):
        self.image_encoder = ByteTransformer(max_bytes=8192)
        self.message_encoder = nn.Embedding(256, 256)  # Encode secret message
        self.fusion = nn.MultiheadAttention(256, num_heads=8)
        self.decoder = ByteTransformerDecoder(max_bytes=8192)

    def forward(self, cover_image_bytes, secret_message_bytes):
        """
        cover_image_bytes: [batch, 8192] - JPEG file bytes
        secret_message_bytes: [batch, message_len] - Secret data to hide
        Returns: stego_image_bytes [batch, 8192] - Modified JPEG with hidden message
        """
        # Encode cover image and secret message
        cover_features = self.image_encoder(cover_image_bytes)
        message_features = self.message_encoder(secret_message_bytes)

        # Fuse message into image
        fused = self.fusion(cover_features, message_features, message_features)

        # Generate stego image bytes
        stego_bytes = self.decoder(fused)  # [batch, 8192, 256]
        return stego_bytes.argmax(dim=-1)
```

### 3.2 Decoder: Extract Message
```python
class ByteStegoDecoder(nn.Module):
    def forward(self, stego_image_bytes):
        """
        stego_image_bytes: [batch, 8192] - JPEG with hidden message
        Returns: extracted_message_bytes [batch, message_len]
        """
        features = self.encoder(stego_image_bytes)
        message_logits = self.message_head(features)
        return message_logits  # Predict secret message
```

### Training Loss
```python
# Multi-task loss
stego_loss = (
    # 1. Visual similarity (stego image should look like cover)
    perceptual_loss(stego_pixels, cover_pixels) +

    # 2. Message recovery (extracted message should match original)
    F.cross_entropy(predicted_message, secret_message) +

    # 3. File validity (stego bytes should decode to valid JPEG)
    validity_loss(stego_bytes)
)
```

---

## 4. Evaluation Metrics

1. **Capacity**: How many bytes can be hidden? (Target: 100-500 bytes in 8KB JPEG)
2. **Visual Quality**: PSNR/SSIM between cover and stego images (Target: >40 dB)
3. **Detection Resistance**: Success rate against steganalysis tools (e.g., StegExpose)
4. **Robustness**: Message recovery after JPEG recompression, resize, crop

---

## 5. Use Cases

1. **Covert Communication**: Hide messages in innocuous photos
2. **Digital Watermarking**: Embed ownership info in images
3. **Data Exfiltration Detection**: Train models to detect hidden data (defensive)
4. **Forensics**: Detect and extract hidden messages from seized files

**Ethical Note**: Dual-use technology - can be used for privacy (good) or evasion (bad). Recommend focus on **detection** rather than **creation** of steganography.

---

## 6. Challenges

1. **File Validity**: Modified bytes must produce valid JPEG (hard constraint)
2. **Detection**: Advanced steganalysis tools (neural networks) getting better
3. **Capacity-Quality Trade-off**: More hidden data → worse visual quality
4. **Robustness**: JPEG recompression may destroy hidden message

---

## 7. Implementation Roadmap (Detection Focus)

**Week 1**: Collect dataset
- Clean images (no steganography)
- Stego images (use existing tools: steghide, outguess, jsteg)

**Week 2**: Implement byte-level steganalysis model
- Train to distinguish clean vs stego JPEG bytes
- Test on unseen steganography tools

**Week 3**: Evaluate
- Detection accuracy, false positive rate
- Compare vs traditional steganalysis (statistical methods)

**Week 4**: Analysis
- What byte patterns indicate steganography?
- Generalization to unknown steganography methods

---

## 8. Success Criteria (Detection Focus)

- [ ] Detect traditional stego methods (LSB, jsteg) with ≥95% accuracy
- [ ] Generalize to unseen stego tools (≥80% accuracy)
- [ ] Low false positive rate (≤5% on clean images)
- [ ] Identify specific file regions containing hidden data

---

## 9. Viability: ⭐⭐ (Low-Medium)

**Strengths**:
- Novel application of byte-level models
- Clear evaluation metrics
- Potential defensive application (steganalysis)

**Challenges**:
- Niche use case (limited audience)
- Ethical concerns (dual-use technology)
- Competing with mature steganography research
- Detection may not be better than existing methods

**Recommendation**: **Low priority** unless specifically interested in security/forensics. Focus on **detection** (defensive) rather than **creation** (offensive). Consider as side project or collaboration with security researchers.

**Timeline**: 3-4 weeks (steganalysis), 6-8 weeks (full stego system)

---

## 10. References

1. **steghide**: Popular steganography tool (https://steghide.sourceforge.net/)
2. **StegExpose**: Steganalysis tool using statistics
3. **Deep Learning for Steganalysis**: Several papers on neural steganalysis
4. **F5, outguess, jsteg**: Classic JPEG steganography algorithms
5. "Hiding Images in Plain Sight" (research on steganography)

---

*Note: This pivot has dual-use implications. Recommend focusing on detection and defense rather than offensive capabilities.*
