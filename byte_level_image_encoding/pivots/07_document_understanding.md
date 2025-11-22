# Pivot 7: Document Understanding (PDF/HTML Byte-Level)

**Research Date**: 2025-11-22
**Viability**: ⭐⭐⭐ (Medium - Complex Format Parsing)

---

## Executive Summary

Process **PDF and HTML files at the byte level** to extract content, understand layout, and answer questions without traditional parsing. Models would learn document structure (headers, tables, images) directly from byte patterns rather than relying on fragile parsing libraries.

**Key Challenge**: PDF/HTML have complex nested structures (compression, fonts, embedded objects) that may be too difficult to learn from raw bytes alone.

---

## 1. Why Byte-Level Document Processing?

### Current Approach (Traditional Parsing)
```
PDF bytes → pdfminer/PyPDF2 → Extract text + layout → NLP model → Answer
                ↑
         Brittle! Fails on complex PDFs, scanned documents, non-standard fonts
```

### Proposed Byte-Level Approach
```
PDF bytes → Byte Model → Direct understanding → Answer
             ↑
    Learn PDF structure (headers, compression, fonts) from data
```

---

## 2. Potential Architectures

### 2.1 Hierarchical PDF Byte Model
```python
class PDFByteModel(nn.Module):
    def __init__(self):
        # Level 1: PDF structure parsing
        self.structure_encoder = ByteTransformer(max_bytes=8192)  # Headers, cross-ref tables

        # Level 2: Content stream parsing
        self.content_encoder = ByteTransformer(max_bytes=32768)  # Text objects, fonts

        # Level 3: Semantic understanding
        self.semantic_encoder = TransformerEncoder(embed_dim=768, depth=12)

    def forward(self, pdf_bytes):
        # Extract structural elements (learned, not rule-based)
        structure_features = self.structure_encoder(pdf_bytes[:8192])

        # Extract content (text, images)
        content_features = self.content_encoder(pdf_bytes)

        # Combine for Q&A
        combined = self.semantic_encoder(content_features + structure_features)
        return combined
```

### 2.2 HTML Byte Model (Simpler)
- HTML is text-based (easier than binary PDF)
- Model learns tags `<div>`, `<p>`, `<img>` from byte patterns
- Potential: Better than tokenization (preserves exact formatting)

---

## 3. Research Questions

1. **Can models learn PDF compression?** (PDF uses Flate/LZW compression)
2. **Font embedding understanding?** (PDFs embed TrueType/Type1 fonts as bytes)
3. **Layout vs content separation?** (Can model distinguish structure from text?)
4. **Scanned PDFs?** (Embedded JPEG images of text - OCR at byte level?)

---

## 4. Feasibility Assessment

### High Feasibility: HTML
- Text-based format
- Simpler structure
- Could beat traditional HTML parsing for robustness

### Medium Feasibility: Simple PDFs
- Text-only PDFs with standard fonts
- Linear structure (no complex graphics)
- Proof-of-concept viable

### Low Feasibility: Complex PDFs
- Scanned documents with embedded images
- Custom fonts and compression
- Interactive forms, JavaScript
- **Likely too complex for end-to-end byte learning**

---

## 5. Hybrid Approach (Most Practical)

```
PDF bytes → Lightweight parser (extract text streams) → Byte-level model → Understanding
              ↑
        Minimal parsing (just decompress), not full layout extraction
```

**Advantage**: Reduce complexity while keeping byte-level processing for text content

---

## 6. Use Cases

1. **Robust PDF Extraction**: Handle PDFs that break traditional parsers
2. **Layout-Aware Q&A**: Answer questions considering document structure
3. **Table Extraction**: Learn table patterns from byte-level structure
4. **HTML Sanitization**: Process untrusted HTML without parsing vulnerabilities
5. **Document Classification**: Classify PDFs by structure (invoices, contracts, papers)

---

## 7. Implementation Roadmap

**Phase 1** (1-2 days): Collect dataset
- arXiv PDFs (scientific papers - relatively standard format)
- HTML pages from CommonCrawl

**Phase 2** (3-5 days): Implement byte-level model
- Start with HTML (simpler)
- Test on webpage classification or Q&A

**Phase 3** (5-7 days): Extend to simple PDFs
- Text-only PDFs from arXiv
- Test on document classification

**Phase 4** (7-10 days): Evaluation
- Compare vs traditional parsing + NLP
- Test robustness on malformed documents

---

## 8. Success Criteria

- [ ] HTML byte model outperforms token-based on robustness
- [ ] Successfully process 80%+ of simple PDFs
- [ ] Demonstrate layout understanding (can identify headers, tables)
- [ ] Advantage on malformed/non-standard documents

---

## 9. Viability: ⭐⭐⭐ (Medium)

**HTML**: High viability (⭐⭐⭐⭐) - simpler, text-based
**Simple PDFs**: Medium viability (⭐⭐⭐) - feasible with hybrid approach
**Complex PDFs**: Low viability (⭐⭐) - too many edge cases, compressed data

**Recommendation**: Start with HTML, extend to simple PDFs if successful. Use hybrid approach (minimal parsing + byte processing) rather than pure byte-level.

**Timeline**: 2-3 weeks (HTML focus), 4-5 weeks (including PDFs)

---

## References

1. LayoutLM, LayoutLMv2: Document understanding with layout (pixel-based, not byte-based)
2. Donut: OCR-free document understanding (image-based)
3. PDFPlumber, PyMuPDF: Traditional parsing libraries (baseline comparison)
