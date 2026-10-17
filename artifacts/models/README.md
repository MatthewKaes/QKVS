# Replayable model checkpoints

These model-only checkpoints omit optimizer state while preserving the exact `ModelConfig` needed
to reconstruct `QKVSDecoder`. Load one with:

```python
import torch

from qkvs.config import ModelConfig
from qkvs.model import QKVSDecoder

saved = torch.load("artifacts/models/three_hop_fourier_mixed.pt", weights_only=True)
model = QKVSDecoder(ModelConfig(**saved["model_config"]))
model.load_state_dict(saved["model"])
model.eval()
```

Included checkpoints:

- `three_hop_block_loss_teacher_safe.pt`: initial strict supervised three-hop baseline;
  SHA-256 `316b9b23d918cb4e1085441c94abd9453e714f6f0bdeda7b0b0fab19c9b0e406`.
- `three_hop_fourier_mixed.pt`: mixed-length address-representation control;
  SHA-256 `66f7afa2823923b21b17f7411d037d3dda1e32fe19cdd64795edd42d53cb6ef1`.
- `span_partial_overlap.pt`: width-4/stride-1 boundary retrieval model;
  SHA-256 `b1c484d180e05754da51b857a007b0ea5326bf2e1caaad35ac729fd535e6452a`.
- `memory_sink_rolling.pt`: learned rolling-memory sink model;
  SHA-256 `6f3cffba24b9a5190bb48bbd4e9190c72a6931630a5e119f52322cc3d8f93bb0`.

They can be regenerated from local training checkpoints with `python -m qkvs.export_checkpoint`.
