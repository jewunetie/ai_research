# Implementation Design

## Overview

This document specifies the technical architecture and implementation strategy for comparing byte-level and pixel-level image encoding on CIFAR-10.

**Goals**:
1. Fair comparison between encoding paradigms
2. Reproducible experiments
3. Modular, extensible codebase
4. Comprehensive evaluation

---

## Project Structure

```
byte_level_image_encoding/
├── CLAUDE.md                    # Project overview
├── RESEARCH.md                  # Literature review
├── ENCODING_SCHEMES.md          # Encoding approaches
├── DESIGN.md                    # Implementation design (this file)
├── IMPLEMENTATION.md            # Detailed implementation plan
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cifar10_bytes.py    # Byte-level dataset
│   │   ├── cifar10_pixels.py   # Pixel-level dataset
│   │   ├── transforms.py       # Format conversions
│   │   └── corruption.py       # Robustness test utilities
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py           # ResNet-18 baseline
│   │   ├── vit.py              # ViT-Tiny baseline
│   │   ├── byteformer.py       # ByteFormer architecture
│   │   ├── hybrid.py           # Hybrid model (future)
│   │   └── layers.py           # Shared components
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   ├── evaluator.py        # Evaluation utilities
│   │   └── metrics.py          # Metric computation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py           # Configuration management
│       ├── logging.py          # Logging utilities
│       └── visualization.py    # Plotting and analysis
│
├── configs/
│   ├── resnet18_pixels.yaml    # ResNet-18 config
│   ├── vit_tiny_pixels.yaml    # ViT-Tiny config
│   ├── byteformer_jpeg.yaml    # ByteFormer JPEG config
│   ├── byteformer_png.yaml     # ByteFormer PNG config
│   └── byteformer_mixed.yaml   # ByteFormer mixed config
│
├── scripts/
│   ├── train.py                # Main training script
│   ├── evaluate.py             # Evaluation script
│   ├── test_robustness.py      # Robustness tests
│   └── analyze_results.py      # Result analysis
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_training.py
│
├── pyproject.toml              # uv project configuration
├── README.md                   # Project README
└── requirements.txt            # Fallback dependencies
```

---

## Architecture Components

### 1. Data Pipeline

#### 1.1 Byte-Level Dataset (`src/data/cifar10_bytes.py`)

```python
class ByteLevelCIFAR10:
    """
    CIFAR-10 dataset returning byte sequences from image files.

    Args:
        root: Dataset root directory
        train: Training or test split
        format: 'jpeg', 'png', or 'mixed'
        jpeg_quality: JPEG compression quality (default: 75)
        max_length: Maximum byte sequence length (default: 8192)
        transform: Additional transforms (corruption, truncation, etc.)
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        format: str = 'mixed',
        jpeg_quality: int = 75,
        max_length: int = 8192,
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        pass

    def image_to_bytes(self, image: PIL.Image, format: str) -> List[int]:
        """Convert PIL Image to byte sequence."""
        buffer = io.BytesIO()
        if format == 'jpeg':
            image.save(buffer, format='JPEG', quality=self.jpeg_quality)
        elif format == 'png':
            image.save(buffer, format='PNG')
        buffer.seek(0)
        byte_array = list(buffer.read())
        return byte_array

    def pad_or_truncate(self, byte_seq: List[int], length: int) -> List[int]:
        """Pad with zeros or truncate to fixed length."""
        if len(byte_seq) < length:
            return byte_seq + [0] * (length - len(byte_seq))
        return byte_seq[:length]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            byte_tensor: LongTensor of shape (max_length,) with values [0, 255]
            label: Integer class label [0, 9]
        """
        pass
```

**Key Design Decisions**:
- **Fixed-length sequences**: Pad/truncate to 8,192 bytes for batching
- **On-the-fly conversion**: Convert images to bytes during training (enables format mixing)
- **Caching option**: Option to pre-convert and cache for faster iteration

---

#### 1.2 Pixel-Level Dataset (`src/data/cifar10_pixels.py`)

```python
class PixelLevelCIFAR10:
    """
    Standard CIFAR-10 with pixel-level preprocessing.

    Args:
        root: Dataset root directory
        train: Training or test split
        transform: Image transformations
        download: Download dataset if not present
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True
    ):
        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download
        )

    @staticmethod
    def get_default_transforms(train: bool = True):
        """Standard CIFAR-10 augmentations."""
        if train:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ])
        else:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)
                )
            ])
```

---

#### 1.3 Corruption Utilities (`src/data/corruption.py`)

```python
class ByteCorruption:
    """Apply byte-level corruptions for robustness testing."""

    def __init__(self, corruption_rate: float = 0.01):
        self.corruption_rate = corruption_rate

    def __call__(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Randomly flip bytes."""
        mask = torch.rand(byte_tensor.shape) < self.corruption_rate
        noise = torch.randint(0, 256, byte_tensor.shape, dtype=torch.long)
        return torch.where(mask, noise, byte_tensor)

class FileTruncation:
    """Truncate byte sequences to simulate incomplete files."""

    def __init__(self, truncation_ratio: float = 0.1):
        self.truncation_ratio = truncation_ratio

    def __call__(self, byte_tensor: torch.Tensor) -> torch.Tensor:
        """Zero out trailing bytes."""
        length = len(byte_tensor)
        keep_length = int(length * (1 - self.truncation_ratio))
        result = byte_tensor.clone()
        result[keep_length:] = 0
        return result
```

---

### 2. Model Architectures

#### 2.1 ByteFormer-CIFAR (`src/models/byteformer.py`)

```python
class ByteFormerCIFAR(nn.Module):
    """
    Byte-level transformer for CIFAR-10.

    Architecture:
        1. Byte embedding (256 vocab → d_model)
        2. Positional encoding (learned)
        3. Strided 1D convolution (8:1 downsampling)
        4. 6 Transformer blocks with shifted window attention
        5. Hierarchical downsampling (after blocks 2, 4)
        6. Global average pooling
        7. Classification head

    Args:
        max_length: Input sequence length (default: 8192)
        d_model: Model dimension (default: 192)
        nhead: Number of attention heads (default: 3)
        num_layers: Number of transformer blocks (default: 6)
        window_size: Window size for shifted attention (default: 128)
        downsample_layers: Layers after which to downsample (default: [2, 4])
        num_classes: Number of output classes (default: 10)
    """

    def __init__(
        self,
        max_length: int = 8192,
        d_model: int = 192,
        nhead: int = 3,
        num_layers: int = 6,
        window_size: int = 128,
        downsample_layers: List[int] = [2, 4],
        num_classes: int = 10
    ):
        super().__init__()

        # Byte embedding: 256 possible byte values
        self.byte_embedding = nn.Embedding(256, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_length, d_model))

        # Initial downsampling: 8:1 strided convolution
        self.initial_downsample = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=8,
            stride=8,
            padding=0
        )
        # After this: 8192 → 1024 tokens

        # Transformer blocks with shifted window attention
        self.transformer_blocks = nn.ModuleList([
            ShiftedWindowTransformerBlock(
                d_model=d_model,
                nhead=nhead,
                window_size=window_size,
                shift=(i % 2 == 1)  # Alternate shifting
            )
            for i in range(num_layers)
        ])

        # Hierarchical downsampling layers
        self.downsample_modules = nn.ModuleDict({
            str(layer): nn.Conv1d(d_model, d_model, kernel_size=2, stride=2)
            for layer in downsample_layers
        })
        # After block 2: 1024 → 512
        # After block 4: 512 → 256

        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_seq: (batch_size, max_length) LongTensor

        Returns:
            logits: (batch_size, num_classes) FloatTensor
        """
        # Embed bytes
        x = self.byte_embedding(byte_seq)  # (B, L, D)
        x = x + self.pos_encoding[:, :byte_seq.size(1), :]

        # Initial downsampling
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.initial_downsample(x)  # (B, D, L/8)
        x = x.transpose(1, 2)  # (B, L/8, D)

        # Transformer blocks with hierarchical downsampling
        for i, block in enumerate(self.transformer_blocks):
            x = block(x)

            # Downsample if specified
            if str(i) in self.downsample_modules:
                x = x.transpose(1, 2)
                x = self.downsample_modules[str(i)](x)
                x = x.transpose(1, 2)

        # Global pooling
        x = self.norm(x)
        x = x.mean(dim=1)  # (B, D)

        # Classification
        logits = self.fc(x)
        return logits
```

---

#### 2.2 Shifted Window Attention (`src/models/layers.py`)

```python
class ShiftedWindowTransformerBlock(nn.Module):
    """
    Transformer block with shifted window attention (1D).

    Reduces complexity from O(n²) to O(n × window_size).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        window_size: int,
        shift: bool = False,
        dropout: float = 0.1
    ):
        super().__init__()
        self.window_size = window_size
        self.shift = shift

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)

        Returns:
            out: (batch_size, seq_len, d_model)
        """
        B, L, D = x.shape

        # Shift if specified
        if self.shift:
            shift_amount = self.window_size // 2
            x = torch.roll(x, shifts=-shift_amount, dims=1)

        # Window partition
        x_windows = self.window_partition(x, self.window_size)  # (B*n_windows, window_size, D)

        # Attention within windows
        attn_out, _ = self.attention(x_windows, x_windows, x_windows)
        attn_out = x_windows + attn_out  # Residual
        attn_out = self.norm1(attn_out)

        # Reverse window partition
        x = self.window_reverse(attn_out, self.window_size, L)  # (B, L, D)

        # Reverse shift
        if self.shift:
            x = torch.roll(x, shifts=shift_amount, dims=1)

        # FFN
        ffn_out = self.ffn(x)
        x = x + ffn_out
        x = self.norm2(x)

        return x

    @staticmethod
    def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
        """Partition sequence into non-overlapping windows."""
        B, L, D = x.shape
        n_windows = L // window_size
        x = x[:, :n_windows * window_size, :]  # Truncate to multiple of window_size
        x = x.view(B, n_windows, window_size, D)
        x = x.view(B * n_windows, window_size, D)
        return x

    @staticmethod
    def window_reverse(x_windows: torch.Tensor, window_size: int, L: int) -> torch.Tensor:
        """Reverse window partitioning."""
        B_windows, _, D = x_windows.shape
        n_windows = B_windows // (L // window_size)  # Infer batch size
        x = x_windows.view(n_windows, -1, window_size, D)
        x = x.view(n_windows, -1, D)
        return x
```

---

#### 2.3 ResNet-18 Baseline (`src/models/resnet.py`)

```python
class ResNet18CIFAR(nn.Module):
    """ResNet-18 adapted for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Use torchvision ResNet-18, modify first conv and remove avgpool
        self.model = torchvision.models.resnet18(weights=None)

        # Adapt for 32×32 images (CIFAR-10)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()  # Remove maxpool for small images
        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```

---

#### 2.4 ViT-Tiny Baseline (`src/models/vit.py`)

```python
class ViTTinyCIFAR(nn.Module):
    """Vision Transformer (Tiny) for CIFAR-10."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        d_model: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embedding = nn.Linear(patch_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=mlp_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=depth
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 3, 32, 32)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Patchify
        p = self.patch_size
        B, C, H, W = x.shape
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
        x = x.contiguous().view(B, C, -1, p, p)  # (B, C, num_patches, p, p)
        x = x.permute(0, 2, 1, 3, 4)  # (B, num_patches, C, p, p)
        x = x.flatten(2)  # (B, num_patches, C*p*p)

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, d_model)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, d_model)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Classification from CLS token
        cls = x[:, 0]
        logits = self.mlp_head(cls)

        return logits
```

---

### 3. Training Pipeline

#### 3.1 Trainer (`src/training/trainer.py`)

```python
class Trainer:
    """General-purpose trainer for all models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        config: Dict[str, Any]
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config

        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return total_loss / len(self.train_loader)

    def validate(self) -> Tuple[float, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def train(self, num_epochs: int):
        """Full training loop."""
        for epoch in range(num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint('best_model.pt')

            # Log
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, '
                  f'Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%')

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }, path)
```

---

### 4. Configuration System

#### 4.1 YAML Configuration (`configs/byteformer_mixed.yaml`)

```yaml
# ByteFormer with mixed format training

experiment_name: "byteformer_cifar10_mixed"
seed: 42

# Data
data:
  dataset: "cifar10"
  root: "./data"
  format: "mixed"  # 'jpeg', 'png', or 'mixed'
  jpeg_quality: 75
  max_length: 8192
  batch_size: 128
  num_workers: 4

# Model
model:
  name: "byteformer"
  d_model: 192
  nhead: 3
  num_layers: 6
  window_size: 128
  downsample_layers: [2, 4]
  num_classes: 10

# Training
training:
  num_epochs: 100
  optimizer: "adamw"
  learning_rate: 0.001
  weight_decay: 0.05
  scheduler: "cosine"
  warmup_epochs: 5

# Logging
logging:
  log_dir: "./logs"
  save_dir: "./checkpoints"
  log_interval: 100
  save_interval: 10
```

---

## Open Questions and Design Decisions

### Question 1: Byte Sequence Padding Strategy
**Options**:
- A) Pad with zeros (0x00)
- B) Pad with special padding token (256)
- C) Use attention mask to ignore padding

**Decision**: **A (Pad with zeros)** - Simplest, zeros are valid bytes that rarely appear at end of valid image files

---

### Question 2: Initial Downsampling Ratio
**Options**:
- A) 4:1 (more information preserved, but longer sequences)
- B) 8:1 (conservative, balanced)
- C) 16:1 (aggressive, faster)

**Decision**: **B (8:1)** - Based on your input "conservative", 8:1 balances information preservation with computational feasibility

---

### Question 3: Window Size for Attention
**Options**:
- A) 64 tokens
- B) 128 tokens (ByteFormer default)
- C) 256 tokens

**Decision**: **B (128 tokens)** - Proven effective in ByteFormer, good balance

---

### Question 4: Format Mixing Strategy
**Options**:
- A) 50% JPEG / 50% PNG
- B) 60% JPEG / 40% PNG (favor compression)
- C) Random per-image per-epoch

**Decision**: **A (50/50)** - Balanced, no format bias

---

### Question 5: Validation Set
**Options**:
- A) Use standard CIFAR-10 test set (10K images)
- B) Split training into train/val (45K/5K)

**Decision**: **A (Use test set)** - CIFAR-10 is well-established, test set is standard validation

---

### Question 6: Target Accuracy
**Options**:
- A) Match ResNet-18 (92-95%)
- B) Within 5% of ResNet-18 (87-90%)
- C) Within 10% + demonstrate robustness advantage

**Decision**: **C (Within 10% + robustness)** - Realistic given computational constraints, focus on unique advantages

---

## Success Criteria

### Must Have (Phase 1-2)
1. ✅ ResNet-18 baseline achieves ≥92% accuracy
2. ✅ ByteFormer trains successfully without errors
3. ✅ ByteFormer achieves ≥80% accuracy (within 15% of baseline)
4. ✅ Clear documentation of accuracy vs. FLOPs tradeoff
5. ✅ Format comparison (JPEG vs PNG vs Mixed)

### Should Have (Phase 3)
6. ✅ Byte corruption robustness test completed
7. ✅ Format generalization test completed
8. ✅ At least one dimension where bytes outperform pixels

### Nice to Have (Phase 4)
9. ⏳ ViT-Tiny baseline implemented
10. ⏳ Hybrid model implemented
11. ⏳ Truncation and compression robustness tests

---

## Next Steps

1. ✅ Define encoding schemes → ENCODING_SCHEMES.md
2. ✅ Design implementation → DESIGN.md (this document)
3. ⏳ Address remaining open questions
4. ⏳ Create detailed implementation plan → IMPLEMENTATION.md
5. ⏳ Set up project structure (uv, dependencies)
6. ⏳ Implement data loaders
7. ⏳ Implement models
8. ⏳ Run experiments
9. ⏳ Analyze results

---

*Last updated: 2025-11-21*
