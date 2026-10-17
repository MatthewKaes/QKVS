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
- `local_memory_general_step400.pt`: fresh-example single-marker write/read model, stopped at step
  400 after convergence; SHA-256
  `465aac968d31c785000d84210fd989cfcfbcfc1a8af335491b79aaa23329d748`.
- `competing_memory_general_step400.pt`: fresh-example four-key, three-distractor write/read model,
  stopped at step 400 after convergence; SHA-256
  `02eafc4038bf1b42940a400ff5be43e1447ddde446d36635976abf9ed44eec0e`.
- `regions8_qkvs_fixed4k_step200.pt`: legacy fixed-4K eight-window geometry diagnostic trained on
  the confounded first dataset; SHA-256
  `a1608042913932c1e5cff46991c4018de6f7314eb5747dbd2bbb0f60e32ca382`.
- `regions8_dense_step400.pt`: legacy dense eight-region control trained on the confounded first
  dataset; SHA-256
  `1ead3eb7f55f79a93d73fce7c60c12adb58490521d5fdadf519009d11c748d72`.
- `regions1_dense_step200.pt`: legacy dense single-region control trained on the confounded first
  dataset; SHA-256
  `f015e615d6579e77dae45e05d481bb3cba36c66a042412f000fc80ae0266f05c`.
- `v2_regions1_dense_step1200.pt`: corrected single-region dense control; it remained at chance and
  did not change predictions under paired pointer counterfactuals; SHA-256
  `936a6e6b982ec85f8eb81c4c6e2fad72f3480a514b2c4cd2959d3d0ec5bfe82e`.
- `v3_regions1_dense_step1200.pt`: single-region dense control with candidate self-address labels,
  still using batch four; it also remained at chance and pointer-insensitive; SHA-256
  `20373b4454d203101d89962f41804aa3aad2fa30007f856493e3b056ce73c27d`.
- `v4_regions8_hierarchical_detached_step1200.pt`: corrected fixed-4K hierarchical detached-router
  baseline; SHA-256
  `faaf56f5a10edc5af3248a7ad9b22923676d6cd424c6f670f51ca5df04ebfaf3`.

They can be regenerated from local training checkpoints with `python -m qkvs.export_checkpoint`.
