# bolt-lab P2: Training Framework Design Spec

## Goal

Add a training framework to bolt-lab that lets method developers run experiments through a uniform interface while keeping full control of their training logic. The framework provides the Method abstract interface, an ExperimentRunner for lifecycle management, and reusable utility components.

## Scope

- **In scope:** Method ABC, ExperimentRunner, utility toolkit (EarlyStopping, Checkpointer, BertBackbone, DataLoader builder, seed utility), method registry, and 3 reference baseline implementations (ADB, GeoID, TLSA).
- **Out of scope:** LLM-based methods (UnLLM, LLM4OpenSSL), Keras methods (SCL), CLI/grid runner (P4), remaining baseline migrations (P3).

## Architecture

```
bolt_lab/
├── methods/
│   ├── __init__.py          # register_method(), load_method(), list_methods()
│   ├── base.py              # Method ABC
│   ├── registry.py          # _METHODS_REGISTRY dict, decorator registration
│   └── _builtin/
│       ├── __init__.py      # auto-import to trigger registration
│       ├── adb.py           # ADB reference implementation
│       ├── geoid.py         # GeoID reference implementation
│       └── tlsa.py          # TLSA reference implementation
├── training/
│   ├── __init__.py          # re-exports
│   ├── runner.py            # ExperimentRunner
│   ├── early_stopping.py    # EarlyStopping utility
│   ├── checkpointer.py      # Checkpointer utility
│   ├── backbone.py          # BertBackbone helper
│   ├── dataloader.py        # make_dataloader(), tokenize helpers
│   └── utils.py             # set_seed(), get_device()
```

## 1. Method Interface (`methods/base.py`)

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from bolt_lab.configs import ExperimentConfig
from bolt_lab.datasets import BoltDataset, BoltSplit


class Method(ABC):
    """Abstract base class for all BOLT methods.

    Each method implements its own training logic. The framework only
    requires: setup, train, predict, and optionally save/load.
    """

    # Subclasses MUST set these
    name: str = ""          # unique method identifier, e.g. "adb"
    task: str = ""          # "gcd" or "openset"

    @abstractmethod
    def setup(
        self,
        config: ExperimentConfig,
        dataset: BoltDataset,
        device: str = "cuda",
    ) -> None:
        """Initialize model, optimizer, data loaders, etc.

        Called exactly once before train(). The method should store
        whatever state it needs from config and dataset.

        Args:
            config: Full experiment configuration.
            dataset: Loaded BoltDataset with all splits populated.
            device: "cuda" or "cpu".
        """

    @abstractmethod
    def train(self) -> dict[str, Any]:
        """Run the full training pipeline.

        This includes ALL training phases: pretrain, train, fine-tune,
        etc. The method has complete control over its training loop.
        For example, ADB runs pretrain(CE) then train(boundary loss)
        inside this single method. GeoID runs pretrain(MLM+CE) then
        train(contrastive+clustering). The framework does not
        distinguish between phases — it's all inside train().

        Returns:
            Summary dict with training metadata (e.g. epochs_run,
            best_dev_metric, training_time). Content is method-specific.
        """

    @abstractmethod
    def predict(self, split: BoltSplit) -> np.ndarray:
        """Produce predictions for a data split.

        For GCD methods: return cluster assignment integers.
        For OSTC methods: return class integers, -1 for "unknown/novel".

        Args:
            split: A BoltSplit (typically dataset.test).

        Returns:
            1D numpy array of predictions, same length as split.
        """

    def save(self, path: Path) -> None:
        """Save model weights and any method state to directory.

        Default: no-op. Override if the method supports checkpointing.
        """

    def load(self, path: Path) -> None:
        """Load model weights and state from directory.

        Default: no-op. Override if the method supports checkpointing.
        """

    def get_info(self) -> dict[str, Any]:
        """Return metadata about the method.

        Default returns name and task. Override to add details like
        model size, backbone type, etc.
        """
        return {"name": self.name, "task": self.task}
```

## 2. Method Registry (`methods/registry.py`)

Same pattern as dataset registry — decorator-based registration.

```python
_METHODS_REGISTRY: dict[str, type[Method]] = {}

def register_method(name: str):
    """Decorator to register a Method subclass."""
    def wrapper(cls):
        _METHODS_REGISTRY[name] = cls
        return cls
    return wrapper

def get_method_class(name: str) -> type[Method]:
    """Get a registered Method class by name."""
    _ensure_builtins()
    if name not in _METHODS_REGISTRY:
        raise ValueError(f"Unknown method '{name}'. Available: {sorted(_METHODS_REGISTRY)}")
    return _METHODS_REGISTRY[name]

def list_methods() -> list[str]:
    """Return sorted list of all registered method names."""
    _ensure_builtins()
    return sorted(_METHODS_REGISTRY)

def _ensure_builtins():
    import bolt_lab.methods._builtin  # noqa: F401
```

The `methods/__init__.py` provides a high-level `load_method()`:

```python
def load_method(name: str, **kwargs) -> Method:
    """Instantiate a method by name."""
    cls = get_method_class(name)
    return cls(**kwargs)
```

## 3. ExperimentRunner (`training/runner.py`)

Orchestrates the full experiment lifecycle, connecting P1 modules to methods.

```python
class ExperimentRunner:
    """Runs a single experiment: load data → setup method → train → predict → evaluate → record."""

    def __init__(
        self,
        config: ExperimentConfig,
        results_manager: ResultsManager | None = None,
        device: str | None = None,
        data_root: str | Path | None = None,
    ):
        self.config = config
        self.results_manager = results_manager
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root

    def run(self, method: Method) -> EvalResult:
        """Execute the full experiment lifecycle.

        Steps:
        1. Check if already done (skip if results exist)
        2. Load dataset with config's split parameters
        3. Call method.setup(config, dataset, device)
        4. Call method.train()
        5. Call method.predict(dataset.test)
        6. Call evaluate(predictions, references, task, known_classes)
        7. Record results via ResultsManager (if provided)
        8. Return EvalResult

        Returns:
            EvalResult with computed metrics.
        """

    def _is_done(self) -> bool:
        """Check if this experiment was already completed."""

    def _load_dataset(self) -> BoltDataset:
        """Load dataset using config's split parameters."""

    def _get_known_class_indices(self, dataset: BoltDataset) -> list[int]:
        """Convert known class names to integer indices for evaluate()."""
```

## 4. Utility Toolkit (`training/`)

### 4.1 EarlyStopping (`training/early_stopping.py`)

```python
class EarlyStopping:
    """Tracks metric improvement and signals when to stop.

    Usage:
        es = EarlyStopping(patience=10, min_delta=0.1, mode="max")
        for epoch in range(max_epochs):
            train_one_epoch()
            score = evaluate_dev()
            if es.step(score):
                break  # no improvement for `patience` epochs
            if es.is_best:
                save_model()
        # es.best_score has the best value seen
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        ...

    def step(self, score: float) -> bool:
        """Update with new score. Returns True if should stop."""

    @property
    def is_best(self) -> bool:
        """Whether the last step() call was a new best."""

    @property
    def best_score(self) -> float:
        """Best score seen so far."""
```

### 4.2 Checkpointer (`training/checkpointer.py`)

```python
class Checkpointer:
    """Save/load PyTorch model checkpoints.

    Usage:
        ckpt = Checkpointer(save_dir="outputs/geoid/models")
        ckpt.save(model, optimizer, epoch=5, tag="best")
        ckpt.load(model, optimizer, tag="best")
    """

    def __init__(self, save_dir: str | Path):
        ...

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        epoch: int = 0,
        tag: str = "latest",
        extra: dict | None = None,
    ) -> Path:
        """Save checkpoint. Returns path to saved file."""

    def load(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        tag: str = "latest",
    ) -> dict:
        """Load checkpoint. Returns extra metadata dict."""

    def exists(self, tag: str = "latest") -> bool:
        """Check if a checkpoint with this tag exists."""
```

### 4.3 BertBackbone (`training/backbone.py`)

```python
class BertBackbone:
    """Convenience wrapper for loading HuggingFace BERT models.

    Handles model loading, feature extraction, and common operations
    like freezing layers.

    Usage:
        bb = BertBackbone("bert-base-uncased", device="cuda")
        features = bb.encode(input_ids, attention_mask)  # [batch, 768]
        bb.freeze(except_layers=["encoder.layer.11", "pooler"])
    """

    def __init__(self, model_name: str, device: str = "cuda"):
        ...

    @property
    def hidden_size(self) -> int:
        """Hidden dimension of the backbone (e.g. 768 for bert-base)."""

    def encode(self, input_ids, attention_mask, token_type_ids=None) -> torch.Tensor:
        """Extract [CLS] embeddings. Returns [batch_size, hidden_size]."""

    def freeze(self, except_layers: list[str] | None = None) -> None:
        """Freeze all params except those matching any pattern in except_layers."""

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""

    def get_tokenizer(self):
        """Return the tokenizer for this backbone."""
```

### 4.4 DataLoader Builder (`training/dataloader.py`)

```python
def make_dataloader(
    split: BoltSplit,
    tokenizer,
    *,
    max_seq_length: int = 128,
    batch_size: int = 32,
    shuffle: bool = False,
    label_map: dict[str, int] | None = None,
) -> DataLoader:
    """Build a PyTorch DataLoader from a BoltSplit.

    Tokenizes texts using the provided HuggingFace tokenizer,
    creates a TensorDataset with (input_ids, attention_mask,
    token_type_ids, labels), and wraps in a DataLoader.

    Args:
        split: BoltSplit with texts and optionally labels.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Max token length (padded/truncated).
        batch_size: Batch size.
        shuffle: Whether to shuffle.
        label_map: Optional label name → int mapping. If None, uses split.labels.

    Returns:
        PyTorch DataLoader yielding (input_ids, attention_mask, token_type_ids, labels).
    """
```

### 4.5 Seed and Device Utilities (`training/utils.py`)

```python
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility (random, numpy, torch, cuda)."""

def get_device(gpu_id: int = 0) -> str:
    """Get device string. Returns 'cuda:N' if available, else 'cpu'."""
```

## 5. Reference Implementations

### 5.1 ADB (`methods/_builtin/adb.py`)

**Pattern:** OSTC, BERT-based, pretrain CE → train boundary loss.

```python
@register_method("adb")
class ADBMethod(Method):
    name = "adb"
    task = "openset"

    def setup(self, config, dataset, device):
        # Load BERT backbone
        # Build train/eval/test dataloaders from dataset splits
        # Initialize BoundaryLoss, optimizer
        # Run pretrain phase (CE on known classes)

    def train(self):
        # Compute centroids from pretrained features
        # Train boundary loss with early stopping on dev F1
        # Store best delta + centroids

    def predict(self, split):
        # Extract features, compute euclidean distance to centroids
        # Reject as unknown (-1) if distance > delta
        # Return class predictions
```

Key components migrated from original:
- `BertForModel` → use `BertBackbone` + custom classifier head
- `PretrainModelManager.train()` → internal pretrain phase in `setup()`
- `ModelManager.train()` → boundary loss training in `train()`
- `open_classify()` → rejection logic in `predict()`
- Early stopping and dev eval preserved

### 5.2 GeoID (`methods/_builtin/geoid.py`)

**Pattern:** GCD, BERT-based, pretrain MLM+CE → train contrastive+clustering.

```python
@register_method("geoid")
class GeoIDMethod(Method):
    name = "geoid"
    task = "gcd"

    def setup(self, config, dataset, device):
        # Load CLBert model (backbone + projection head + classifier)
        # Build labeled/semi/eval/test dataloaders
        # Run pretrain (InternalPretrainModelManager equivalent)
        # Initialize memory bank, feature bank, pseudo labels

    def train(self):
        # For each epoch:
        #   Compute neighbor indices via memory bank
        #   Build neighbor dataset
        #   Train with contrastive + cross-entropy + KL losses
        #   Update feature bank, pseudo labels
        #   KMeans clustering for pseudo labels
        #   Early stopping on dev clustering ACC

    def predict(self, split):
        # Extract features, run KMeans, return cluster assignments
```

Key components:
- `CLBert` model class (backbone + projection head + ETF classifier)
- `MemoryBank`, `NeighborsDataset` — method-specific, kept internal
- Two-phase: pretrain (MLM+CE) → train (contrastive+clustering)
- `view_generator` for RTR augmentation

### 5.3 TLSA (`methods/_builtin/tlsa.py`)

**Pattern:** GCD, BERT-based, pretrain CLIP → train semi-supervised EM.

```python
@register_method("tlsa")
class TLSAMethod(Method):
    name = "tlsa"
    task = "gcd"

    def setup(self, config, dataset, device):
        # Create TT-CLIP model (text encoder + label encoder)
        # Build dataloaders for phase 1 (text-label pairs) and phase 2 (semi-supervised)
        # Initialize CLIP loss, cluster manager

    def train(self):
        # Phase 1: supervised warm-up with CLIP loss
        #   Early stopping on dev k-acc
        # Phase 2: semi-supervised EM training
        #   E-step: cluster features → pseudo labels
        #   M-step: train on pseudo labels + labeled data
        #   Early stopping on dev clustering metrics

    def predict(self, split):
        # Extract features, run clustering, return assignments
```

Key components:
- `create_ttclip_model` — TT-CLIP model with text/label encoders
- `create_loss_function` — CLIP-style loss
- `create_cluster_manager` — KMeans/spectral clustering
- Two-phase: supervised warm-up → semi-supervised EM

## 6. Dependencies

P2 adds `torch` and `transformers` as required dependencies (moved from optional):

```toml
dependencies = [
  "pyyaml>=6.0",
  "numpy>=1.23",
  "pandas>=2.0",
  "scikit-learn>=1.3",
  "tabulate>=0.9",
  "torch>=2.0",
  "transformers>=4.35",
]
```

Note: `torch` becomes a core dependency since the training framework is PyTorch-based. Methods using other frameworks (Keras/TF) can declare their own dependencies.

## 7. Data Flow

```
ExperimentConfig + Method name
        │
        ▼
ExperimentRunner.run(method)
        │
        ├─ load_dataset(config.dataset, **config.split)  → BoltDataset
        │
        ├─ method.setup(config, dataset, device)
        │    └── internally builds DataLoaders, model, optimizer
        │
        ├─ method.train()
        │    └── pretrain → train (method controls entirely)
        │
        ├─ method.predict(dataset.test) → np.ndarray predictions
        │
        ├─ evaluate(predictions, references, task, known_classes) → EvalResult
        │
        └─ results_manager.record(result, method=..., dataset=..., ...)
```

## 8. Testing Strategy

- **Unit tests for utilities:** EarlyStopping, Checkpointer, set_seed, make_dataloader
- **Integration test for runner:** Mock Method that returns fixed predictions, verify full lifecycle
- **Smoke test for reference methods:** ADB/GeoID/TLSA on banking dataset with KCR=0.5, verify metrics are reasonable (not perfect, just non-zero)
- **Registry tests:** register_method, list_methods, load_method

## 9. File Size Guidance

Each reference implementation (ADB, GeoID, TLSA) will be a single file containing:
- The Method subclass
- Model architecture class(es) specific to that method
- Any method-specific loss functions or utilities

Target: each file should be under 400 lines. If a method requires more, split its model architecture into a separate `_models/` file.
