# VLA Weight Loading, Step 1

The first thing to learn from OpenPI is the loading pattern, not Gemma itself.
Their pi0 loader initializes the full target model, overlays compatible
pretrained arrays, and leaves new robotics-specific weights at their initialized
values.

In PyTorch terms:

```python
from robot_control_policies.weights import load_pretrained_weights

model = MyPolicy(...)
report = load_pretrained_weights(
    model,
    "path/to/checkpoint.pt",
    allow_missing_regex=".*",
)
print(report.summary())
```

This does four concrete things:

1. Builds the target model first, so every desired parameter already exists.
2. Loads only checkpoint tensors whose names exist in the target model.
3. Rejects tensors with incompatible shapes.
4. Fills missing tensors from the target model's random initialization.

That fourth step is what makes partial loading useful for VLAs. For example, a
future policy might load a language backbone from Gemma while keeping the action
head randomly initialized:

```text
language_backbone.*  <- pretrained
vision_backbone.*    <- pretrained or random
action_expert.*      <- random at first
flow_head.*          <- random at first
```

The model architecture still controls whether loading succeeds. A tensor from a
checkpoint can only be copied when both the parameter name and tensor shape match
the target model.

## How This Maps To OpenPI

OpenPI's pi0 setup uses a PaliGemma/Gemma backbone for image and language tokens
plus a separate action expert. Their loader keeps the matching PaliGemma weights
and fills the missing action-expert weights from initialization.

The same design can be built in this repo in small steps:

1. Add backbone modules behind simple token interfaces.
2. Give each backbone stable names such as `vision_backbone`,
   `language_backbone`, and `action_expert`.
3. Load pretrained weights into those matching names.
4. Keep policy-specific heads randomly initialized until you have a pi0
   checkpoint for them.

The important habit is to inspect the report after every load. Large numbers of
missing or shape-mismatched tensors usually mean the architecture or naming does
not match the checkpoint you thought you were loading.
