# Pivot 4: Audio/Video Byte-Level Models

**Research Date**: 2025-11-22
**Searches Conducted**: 10 targeted web searches
**Papers Analyzed**: 40+ publications from 2022-2025
**Viability**: ⭐⭐⭐⭐ (High - Established with Strong Growth)

---

## Executive Summary

This pivot explores extending **byte-level encoding approaches from images to audio and video modalities**. Neural audio codecs (EnCodec, DAC, SoundStream) have achieved 90-200× compression via discrete token representations, enabling generative models like MusicGen and AudioLM. Video compression is advancing through learned neural codecs that rival or surpass H.266/VVC. The key insight: **discrete audio/video tokens enable unified multimodal modeling** similar to text, opening pathways for byte-level processing across all media types.

**Key Finding**: Audio neural codecs achieve **state-of-the-art compression** (DAC: 90×, EnCodec: 65×) with residual vector quantization (RVQ), while learned video codecs (DCVC-FM) match traditional H.266/VVC performance. Cross-modal byte prediction (bGPT) demonstrates positive transfer between images and audio, suggesting **unified byte-level representations** are viable across modalities.

---

## 1. State-of-the-Art Models (2024-2025)

### 1.1 Neural Audio Codecs

**DAC (Descript Audio Codec)** - NeurIPS 2023, Active in 2024
- **Compression**: **90× compression** of 44.1 kHz audio → 8 kbps tokens
- **Architecture**: Fully convolutional encoder-decoder with RVQ (Residual Vector Quantization)
- **Universal Model**: Single model compresses speech, music, and environmental audio
- **Sample Rates**: Supports 44.1kHz, 24kHz, 16kHz (mono/stereo)
- **Key Innovation**: Combines high-fidelity audio generation techniques with improved VQ from image domain, plus enhanced adversarial and reconstruction losses
- **Status**: **State-of-the-art as of 2024**, integrated into Hugging Face Transformers (August 2024)
- **GitHub**: 1,500+ stars, active development

**EnCodec (Meta AI)** - October 2022, Improvements in 2024
- **Compression**: **65-160× compression** at 1.5-24 kbps bandwidth
- **Architecture**: Streaming convolutional encoder-decoder with 3-stage RVQ
- **Components**:
  1. Encoder: Audio → latent representation
  2. RVQ: Compress latents to discrete tokens
  3. Decoder: Reconstruct time-domain audio from tokens
- **Real-Time**: Streaming capable, low latency
- **Recent Advances (2024)**: **ERVQ (Enhanced RVQ)** - improved codebook optimization
  - Intra-codebook and inter-codebook optimization
  - Consistent improvements across HiFi-Codec, APCodec, EnCodec
  - Variable bitrate support (VRVQ) for adaptive compression
- **Applications**: AudioLM, MusicGen, AudioGen

**SoundStream (Google)** - 2021, Foundational
- **Architecture**: Early RVQ-based neural audio codec
- **Influence**: Inspired EnCodec and DAC architectures
- **Status**: Foundational work, superseded by more recent codecs

**HiFi-Codec** - May 2023
- **Innovation**: Group-residual vector quantization for high fidelity
- **Performance**: Improved audio quality vs standard RVQ
- **Integration**: Compatible with ERVQ enhancements (2024)

### 1.2 Speech Models (Waveform-Level, Not Strictly Byte-Level)

**Whisper (OpenAI)** - 2022, Updated 2024
- **Architecture**: Encoder-decoder transformer with **byte-pair encoding (BPE)** tokenizer
- **Tokenization**: Uses GPT-2 BPE tokenizer (English) or multilingual retrained vocabulary
- **Token Space**: 51,865 tokens (byte-level BPE, outputs Unicode byte codes)
- **Performance**: Robust speech recognition across 99 languages
- **Recent Update (May 2024)**: GPT-4o integration for multimodal ASR
- **Note**: Operates on **mel-spectrograms** (not raw bytes), but uses byte-level text tokenization

**Wav2vec 2.0 & HuBERT (Meta FAIR)** - 2020-2021, Updated 2024
- **Architecture**: Self-supervised learning on raw waveforms (not bytes)
- **Wav2vec 2.0**: Contrastive learning on masked waveform patches
- **HuBERT**: Masked language modeling with offline clustering
- **Recent**: Wav2Vec2-BERT 2.0 (Meta, 2024) - pretrained on **4.5M hours** of audio
- **Visual Grounding (Sept 2025)**: Audio-visual SSL reduces multilingual gap from 31.5% → 8.04%
- **Limitation**: Waveform-level (continuous), not byte/token-level

### 1.3 Learned Video Codecs

**DCVC-DC & DCVC-FM (Deep Contextual Video Compression)** - 2024
- **Performance**: Superior under **low delay** conditions vs traditional codecs
- **Comparison**: Competitive with H.266/VVC (Versatile Video Coding)
- **Architecture**: Deep learning-based, frame-level adaptive rate control via dynamic neural networks
- **Use Case**: Real-time streaming, video conferencing

**H.266/VVC (Traditional Codec Baseline)** - 2020, Maturing 2025-2026
- **Compression**: 40-50% improvement over H.265/HEVC
- **Speed**: 10-20% faster than AV1
- **Adoption**: Broader availability expected late 2025 to early 2026
- **FFmpeg Support**: Official decoders as of July 2025 (v7.1)
- **Trade-off**: High encoding/decoding complexity

**JVET ECM (Enhanced Compression Model)** - Ongoing
- **Performance**: **Best overall coding performance** among tested codecs (including VVC, AV1, neural)
- **Trade-off**: Very high encoding/decoding complexity
- **Status**: Research/development stage

### 1.4 Generative Audio Models (Using Discrete Tokens)

**AudioLM (Google)** - 2022
- **Architecture**: Hierarchical tokenization with semantic + acoustic tokens
- **Codec**: Uses SoundStream or EnCodec for discrete tokens
- **Approach**: Transformer-based sequence modeling
- **Capability**: High-quality audio continuation and generation

**MusicGen (Meta AI)** - 2023
- **Architecture**: Transformer operating on EnCodec discrete tokens
- **Capability**: Text-to-music generation
- **Token Rate**: ~50-75 tokens/sec (EnCodec compressed)
- **Status**: State-of-the-art music generation

**AudioGen (Meta AI)** - 2023
- **Focus**: General audio and sound effects generation
- **Architecture**: Similar to MusicGen, uses EnCodec tokens
- **Capability**: Text-to-audio (non-speech, non-music)

### 1.5 Cross-Modal Byte Models

**bGPT (Byte GPT)** - February 2024 (from earlier pivot research)
- **Key Finding**: **Positive transfer** from ImageNet bytes → audio tasks
- **Negative Transfer**: Text bytes → audio/images (text has distinct byte patterns)
- **Implication**: **Shared byte patterns** exist between images and audio at file level
- **Approach**: Next-byte prediction on raw file bytes
- **Performance**: Comparable to specialized models on audio/visual data without modality-specific designs

---

## 2. Technical Architecture Patterns

### 2.1 Residual Vector Quantization (RVQ)

**Core Concept**: Iterative quantization of residuals
```python
class ResidualVectorQuantizer:
    def __init__(self, num_codebooks=8, codebook_size=1024, dim=256):
        self.codebooks = nn.ModuleList([
            VectorQuantize(dim, codebook_size) for _ in range(num_codebooks)
        ])

    def forward(self, x):
        """
        x: [batch, dim, time] - latent representation
        Returns: quantized, codes, commitment_loss
        """
        quantized = torch.zeros_like(x)
        codes = []
        residual = x
        total_loss = 0

        for codebook in self.codebooks:
            # Quantize residual
            q, indices, loss = codebook(residual)
            quantized += q
            codes.append(indices)
            total_loss += loss

            # Update residual
            residual = residual - q

        return quantized, torch.stack(codes, dim=1), total_loss
```

**Benefits**:
- **Progressive refinement**: Each codebook reduces quantization error
- **Variable bitrate**: Use fewer codebooks for lower bitrate (VRVQ, 2024)
- **High fidelity**: 8-16 codebooks achieve near-lossless quality

**Used by**: EnCodec, DAC, SoundStream, HiFi-Codec

### 2.2 Neural Audio Codec Architecture (EnCodec/DAC Pattern)

```python
class NeuralAudioCodec(nn.Module):
    def __init__(self, channels=1, sample_rate=24000, n_codebooks=8):
        # Encoder: Raw audio → latent
        self.encoder = ConvolutionalEncoder(
            in_channels=channels,
            out_channels=256,
            ratios=[8, 5, 4, 2],  # Total downsampling: 320×
            activation='ELU'
        )

        # Quantizer: Latent → discrete tokens
        self.quantizer = ResidualVectorQuantizer(
            num_codebooks=n_codebooks,
            codebook_size=1024,
            dim=256
        )

        # Decoder: Tokens → reconstructed audio
        self.decoder = ConvolutionalDecoder(
            in_channels=256,
            out_channels=channels,
            ratios=[2, 4, 5, 8]
        )

        # Discriminator (for adversarial training)
        self.discriminator = MultiScaleDiscriminator()

    def encode(self, audio):
        """audio: [batch, channels, time]"""
        latent = self.encoder(audio)
        quantized, codes, vq_loss = self.quantizer(latent)
        return codes, vq_loss  # codes: [batch, n_codebooks, time/320]

    def decode(self, codes):
        """codes: [batch, n_codebooks, time/320]"""
        # Lookup codes in quantizer
        quantized = self.quantizer.decode(codes)
        audio = self.decoder(quantized)
        return audio

    def forward(self, audio):
        codes, vq_loss = self.encode(audio)
        reconstruction = self.decode(codes)

        # Adversarial + reconstruction loss
        adv_loss = self.discriminator(reconstruction, audio)
        recon_loss = F.l1_loss(reconstruction, audio)

        total_loss = recon_loss + 0.1 * vq_loss + 0.01 * adv_loss
        return reconstruction, total_loss
```

**Key Parameters**:
- **Downsampling**: 320× (audio) vs 16× (images) - audio requires less spatial compression
- **Codebooks**: 8-16 (audio) vs 1-2 (images) - audio needs finer quantization
- **Sample Rate**: 24-48 kHz → 75-150 tokens/sec

### 2.3 Byte-Level Audio Processing (Hypothetical Extension)

**Concept**: Apply byte-level encoding to audio file bytes (similar to ByteFormer for images)

```python
class ByteAudioEncoder(nn.Module):
    def __init__(self, max_bytes=16384, embed_dim=256):
        # Audio files: WAV ~100KB, MP3 ~50KB for 10 sec
        self.byte_embedding = nn.Embedding(256, embed_dim)
        self.pos_encoding = PositionalEncoding(max_bytes, embed_dim)

        # Efficient attention for long sequences
        self.transformer = TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=8,
            nhead=8,
            use_flash_attention=True  # Critical for 16K bytes
        )

        # Classification head (e.g., audio event classification)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, audio_bytes):
        """
        audio_bytes: [batch, seq_len] - LongTensor of byte values 0-255
        seq_len: ~16,384 for 10 sec MP3 at 128 kbps
        """
        # Embed bytes
        embedded = self.byte_embedding(audio_bytes)  # [batch, seq_len, embed_dim]
        embedded = self.pos_encoding(embedded)

        # Encode
        features = self.transformer(embedded)  # [batch, seq_len, embed_dim]

        # Global pooling + classify
        pooled = features.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
```

**Challenges**:
- **Very long sequences**: 10 sec MP3 @ 128 kbps = ~160,000 bytes (20× longer than images)
- **Format complexity**: MP3/AAC use complex encoding (Huffman, MDCT)
- **Sequential nature**: Audio is inherently temporal (unlike 2D images)

**Solutions**:
- **Hierarchical processing**: MEGABYTE-style patching
- **Downsampling**: Strided convolutions (16:1 or 32:1)
- **Efficient attention**: Flash Attention, Linformer, or Performer

---

## 3. Performance Benchmarks

### 3.1 Audio Codec Compression Ratios

| Codec | Input | Output | Compression | Bitrate | Quality (ViSQOL) |
|-------|-------|--------|-------------|---------|------------------|
| **DAC** | 44.1 kHz | ~75 tokens/sec | **90×** | 8 kbps | ~4.5 (high) |
| **EnCodec** | 24 kHz | ~50 tokens/sec | **65×** | 6 kbps | ~4.3 (high) |
| **SoundStream** | 24 kHz | ~40 tokens/sec | **60×** | 3 kbps | ~4.0 (medium) |
| **MP3** | 44.1 kHz | - | ~10× | 128 kbps | ~4.0 (reference) |
| **Opus** | 48 kHz | - | ~12× | 64 kbps | ~4.2 (high) |

**Key Insight**: Neural codecs achieve **6-9× better compression** than traditional MP3/Opus while maintaining comparable quality.

### 3.2 Video Codec Performance (BD-rate vs H.265/HEVC)

| Codec | BD-rate Savings | Encoding Complexity | Decoding Complexity | Status |
|-------|----------------|---------------------|---------------------|--------|
| **H.266/VVC** | 40-50% | Very High | High | Maturing 2025-26 |
| **JVET ECM** | **~55%** | Extreme | Extreme | Research |
| **DCVC-FM** | ~35-40% | High | Medium | Production-ready |
| **DCVC-DC** | ~30-35% | Medium | Medium | Production-ready |
| **AV1** | 30% | High | Medium | Widely adopted |

**Key Insight**: Learned video codecs (DCVC) approaching traditional codec performance with potential for real-time deployment.

### 3.3 Speech Recognition Performance

| Model | Architecture | Params | WER (LibriSpeech) | Languages |
|-------|-------------|--------|-------------------|-----------|
| **Whisper Large V3** | Transformer + BPE | 1.55B | ~2.5% | 99 |
| **Wav2Vec2-BERT 2.0** | Waveform SSL | 600M | ~2.0% | English |
| **HuBERT Large** | Waveform SSL + clustering | 300M | ~2.3% | English |
| **wav2vec 2.0 XLS-R** | Multilingual SSL | 2B | ~3.0% | 128 |

**Key Insight**: Byte-level text tokenization (Whisper BPE) enables multilingual coverage, but waveform-level models (wav2vec2) achieve best monolingual performance.

---

## 4. Research Opportunities for Byte-Level Encoding

### 4.1 Byte-Level Audio File Classification

**Concept**: Classify audio directly from MP3/WAV file bytes (no decompression)

**Experiment Design**:
```
Dataset: AudioSet (632K audio clips, 527 sound classes)
Formats: MP3 (128 kbps), WAV (16-bit PCM), FLAC

Baselines:
1. Waveform Model (wav2vec 2.0 + linear probe)
2. Spectrogram CNN (ResNet-50 on mel-spectrograms)

Byte-Level Approach:
1. ByteAudio-Former: Transformer on audio file bytes
2. Byte-MAE: Masked autoencoding of audio bytes (pre-train then fine-tune)

Metrics:
- Mean Average Precision (mAP) on AudioSet
- Inference time (ms/clip)
- Robustness to bitrate changes (64, 128, 256 kbps)
- Format generalization (MP3 → AAC, FLAC)
```

**Hypothesis**: Byte-level models will have **lower clean accuracy** but **better robustness** to:
- Codec changes (MP3 → AAC → Opus)
- Bitrate variations
- File corruption
- Truncated files

### 4.2 Cross-Modal Byte Pre-training (Image + Audio)

**Concept**: Pre-train unified byte model on images + audio, then fine-tune on each modality

**Architecture** (inspired by bGPT and EvaByte):
```python
class UnifiedByteModel(nn.Module):
    def __init__(self, max_bytes=16384, embed_dim=512):
        # Shared byte embedding
        self.byte_embedding = nn.Embedding(256, embed_dim)

        # Modality-specific position encodings
        self.image_pos = PositionalEncoding2D(max_bytes, embed_dim)
        self.audio_pos = PositionalEncoding1D(max_bytes, embed_dim)

        # Shared transformer backbone
        self.transformer = TransformerEncoder(embed_dim, depth=12)

        # Modality-specific heads
        self.image_head = nn.Linear(embed_dim, 1000)  # ImageNet classes
        self.audio_head = nn.Linear(embed_dim, 527)   # AudioSet classes

    def forward(self, bytes, modality='image'):
        embedded = self.byte_embedding(bytes)

        if modality == 'image':
            embedded = self.image_pos(embedded)
            logits = self.image_head(self.transformer(embedded).mean(1))
        else:  # audio
            embedded = self.audio_pos(embedded)
            logits = self.audio_head(self.transformer(embedded).mean(1))

        return logits
```

**Training Procedure**:
1. **Stage 1**: Pre-train on next-byte prediction (50% images, 50% audio)
2. **Stage 2**: Fine-tune on classification (image: ImageNet, audio: AudioSet)
3. **Evaluation**: Compare vs modality-specific models

**Research Questions**:
- Does image byte pre-training help audio tasks? (bGPT showed positive transfer)
- Are shared byte patterns sufficient for both modalities?
- What's the optimal mixing ratio during pre-training?

### 4.3 Byte-Level Video Understanding

**Concept**: Process video file bytes for action recognition or video classification

**Challenges**:
- **Massive sequences**: 1 min H.264 video @ 1 Mbps = 7.5 MB = **7,500,000 bytes**
- **Temporal structure**: Keyframes, P-frames, B-frames (complex inter-dependencies)
- **Codec complexity**: H.264/H.265 use motion compensation, loop filtering, etc.

**Hierarchical Approach** (MEGABYTE-inspired):
```python
class ByteVideoModel:
    def __init__(self):
        # Level 1: Patch bytes into chunks (e.g., 1024 bytes per chunk)
        self.patch_encoder = ByteTransformer(max_bytes=1024, embed_dim=256)

        # Level 2: Sequence of patch embeddings
        self.global_encoder = TransformerEncoder(embed_dim=256, depth=8)

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, video_bytes):
        """
        video_bytes: [batch, ~1M bytes] for 1 sec video
        """
        # Split into chunks of 1024 bytes
        chunks = video_bytes.view(batch, -1, 1024)  # [batch, ~1000, 1024]

        # Encode each chunk
        chunk_embeddings = []
        for i in range(chunks.size(1)):
            emb = self.patch_encoder(chunks[:, i])  # [batch, 256]
            chunk_embeddings.append(emb)

        # Stack and encode sequence of chunks
        chunk_seq = torch.stack(chunk_embeddings, dim=1)  # [batch, ~1000, 256]
        global_features = self.global_encoder(chunk_seq)

        # Classify
        logits = self.classifier(global_features.mean(dim=1))
        return logits
```

**Feasibility**: **Medium** - requires significant computational resources but theoretically tractable

### 4.4 Format-Agnostic Audio Codec Learning

**Concept**: Train byte-level model to classify audio content regardless of codec format

**Setup**:
```
Training Data: Same audio clips encoded in multiple formats
- MP3 (128 kbps)
- AAC (128 kbps)
- Opus (64 kbps)
- FLAC (lossless)
- WAV (uncompressed)

Task: Audio event classification (e.g., ESC-50, 50 classes)

Models:
1. Baseline: Waveform CNN (requires decoding)
2. Byte-Specific: Separate models per format
3. Byte-Unified: Single model trained on mixed formats
4. Byte-Codec-Aware: Multi-task learning (classify + predict format)
```

**Expected Outcome**: Byte-unified model achieves **80-90% of baseline accuracy** but works on **all formats** without format-specific logic.

---

## 5. Viability Assessment

### 5.1 Strengths

1. **Proven Neural Codecs**: DAC, EnCodec achieve SOTA compression with high fidelity
2. **Discrete Token Ecosystem**: AudioLM, MusicGen demonstrate viability of token-based audio modeling
3. **Cross-Modal Transfer**: bGPT shows positive transfer between image and audio bytes
4. **Waveform SSL Success**: Wav2vec2, HuBERT prove self-supervised learning works for audio
5. **Video Codec Progress**: DCVC-FM approaching traditional codec performance

### 5.2 Challenges

1. **Sequence Length**: Audio files 10-20× longer than images (160KB vs 8KB)
2. **Video Complexity**: Video files 100-1000× longer (1-10 MB for 1 min)
3. **Codec Complexity**: MP3/AAC use Huffman coding, making byte patterns less structured
4. **Temporal Dependencies**: Audio/video have strong temporal structure vs spatial (images)
5. **Computational Cost**: O(n²) attention infeasible for millions of bytes

### 5.3 Recommended Approach

**Phase 1: Audio (10-20 sec clips)**
- Start with short audio clips (AudioSet, ESC-50)
- Use hierarchical patching (MEGABYTE approach)
- Focus on robustness and format generalization

**Phase 2: Cross-Modal (Image + Audio)**
- Pre-train unified byte model on CIFAR-10 images + short audio
- Measure transfer learning benefits
- Compare vs modality-specific baselines

**Phase 3: Video (if feasible)**
- Use very short clips (1-5 sec)
- Leverage keyframe detection to reduce redundancy
- Hierarchical multi-level encoding

**Skip if**: Computational resources insufficient or audio experiments show no advantage over waveform models

---

## 6. Implementation Roadmap

### Phase 1: Neural Audio Codec Baseline (3-5 days)

```bash
# Integrate existing codec
pip install encodec dac

# Evaluate on classification
python eval_codec_features.py --model encodec --dataset audioset_short

# Expected: ~85-90% accuracy using codec features
```

### Phase 2: Byte-Level Audio (5-7 days)

```python
# Implement ByteAudio model
python train_byte_audio.py \
  --dataset esc50 \
  --max_bytes 16384 \
  --format mp3 \
  --downsampling 16

# Expected: 70-80% accuracy (vs 90-95% waveform baseline)
```

### Phase 3: Cross-Modal Pre-training (7-10 days)

```python
# Pre-train on images + audio
python pretrain_unified_bytes.py \
  --image_data cifar10 \
  --audio_data esc50 \
  --mixing_ratio 0.5

# Fine-tune on each modality
python finetune_image.py --checkpoint unified_bytes.pth
python finetune_audio.py --checkpoint unified_bytes.pth

# Measure transfer benefits
```

### Phase 4: Evaluation (2-3 days)

**Metrics**:
- Classification accuracy (clean data)
- Robustness: bitrate changes, format changes, corruption
- Efficiency: FLOPs, latency, memory
- Format generalization: MP3 → AAC, WAV

---

## 7. Expected Outcomes

| Approach | Audio Accuracy | Format Robustness | Inference Speed | Novelty |
|----------|----------------|-------------------|-----------------|---------|
| Waveform CNN | 90-95% | Low (format-specific) | Fast (GPU) | Low |
| Wav2vec2 features | 92-96% | Medium | Medium | Low |
| Neural codec features | 85-90% | Medium | Fast | Medium |
| **Byte-level audio** | **70-85%** | **High** | **Slow** | **High** |
| **Cross-modal bytes** | **75-90%** | **Very High** | **Medium** | **Very High** |

**Key Insights**:
1. Byte-level trades accuracy for robustness and format generalization
2. Cross-modal pre-training likely provides 5-10% boost over byte-only
3. Video byte-level modeling feasible only for very short clips (<5 sec)

---

## 8. Success Criteria

### Must-Have
- [ ] Evaluate DAC/EnCodec features on audio classification
- [ ] Implement byte-level audio model (MP3/WAV)
- [ ] Achieve ≥70% on ESC-50 or similar benchmark
- [ ] Demonstrate format robustness (MP3 → AAC)

### Should-Have
- [ ] Cross-modal image+audio pre-training
- [ ] Show positive transfer (≥5% improvement)
- [ ] Format generalization experiments
- [ ] Compression ratio vs accuracy analysis

### Nice-to-Have
- [ ] Byte-level video model (short clips)
- [ ] Publish unified byte model and weights
- [ ] Real-time inference demo

---

## 9. Conclusion

Audio and video byte-level modeling is **highly viable** with established neural codec infrastructure (DAC, EnCodec, DCVC-FM) and proven discrete token ecosystems (AudioLM, MusicGen). The main challenges are **sequence length** and **codec complexity**, but hierarchical architectures (MEGABYTE, BLT) and cross-modal transfer (bGPT) provide clear solution paths.

**Viability: ⭐⭐⭐⭐ (High)** - Strong foundation with neural codecs, cross-modal transfer demonstrated, clear implementation path. Recommended as a natural extension of image byte-level encoding with unique advantages in format robustness and multimodal unification.

**Timeline**: 3-4 weeks (audio focus), 5-6 weeks (including cross-modal)
**GPU Hours**: ~100-150 hours (A100)
**Storage**: ~20 GB (datasets, checkpoints)

---

## 10. Key References

1. **DAC**: Kumar et al., "High-Fidelity Audio Compression with Improved RVQGAN", NeurIPS 2023
2. **EnCodec**: Défossez et al., "High Fidelity Neural Audio Compression", arXiv 2022
3. **ERVQ**: "Enhanced Residual Vector Quantization for Neural Audio Codecs", Oct 2024
4. **Whisper**: Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", 2022
5. **Wav2vec 2.0**: Baevski et al., "wav2vec 2.0: A Framework for Self-Supervised Learning", NeurIPS 2020
6. **DCVC-FM**: "Learned Rate Control for Frame-Level Adaptive Neural Video Compression", Aug 2024
7. **H.266/VVC**: "Versatile Video Coding (VVC)", ISO/IEC 23090-3
8. **bGPT**: "Beyond Language Models: Byte Models are Digital World Simulators", Feb 2024
9. **AudioLM**: Borsos et al., "AudioLM: A Language Modeling Approach to Audio Generation", 2022
10. **MusicGen**: Copet et al., "Simple and Controllable Music Generation", 2023

---

*Research compiled: 2025-11-22*
*Status: High viability - recommended for exploration*
*Best suited for: Audio classification, cross-modal learning, format robustness*
