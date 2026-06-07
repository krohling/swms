from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from .configuration_paligemma_wm import PaliGemmaWMConfig
from .modeling_paligemma_wm import (
    PaliGemmaWMForConditionalGeneration,
    PaliGemmaWMModel,
    PaliGemmaWMPreTrainedModel,
)
from .processing_paligemma_wm import PaliGemmaWMProcessor

# Register with the HF auto classes. `exist_ok=True` makes these calls idempotent,
# so re-importing this module (e.g. under multi-process launchers, notebook reloads,
# or test suites) does not raise.
AutoConfig.register(PaliGemmaWMConfig.model_type, PaliGemmaWMConfig, exist_ok=True)
# `AutoModel` must point at the base model (no LM head); `AutoModelForImageTextToText`
# owns the generation-capable head, matching upstream PaliGemma's mapping.
AutoModel.register(PaliGemmaWMConfig, PaliGemmaWMModel, exist_ok=True)
AutoModelForImageTextToText.register(
    PaliGemmaWMConfig, PaliGemmaWMForConditionalGeneration, exist_ok=True
)
AutoProcessor.register(PaliGemmaWMConfig, PaliGemmaWMProcessor, exist_ok=True)

# Register checkpoint conversion mapping so `from_pretrained` can remap the legacy
# key layout used by published checkpoints (e.g. `jacob3333/paligemma-3b-mix-224-wm`,
# which was saved before the PaliGemmaModel + ForConditionalGeneration split and has
# keys like `vision_tower.*`, `language_model.model.*`, and `multi_modal_projector.*`)
# to the current layout where everything except `lm_head` sits under `model.*`.
#
# This mirrors the entries that upstream transformers ships for `model_type == "paligemma"`.
# It is only available on transformers forks that expose `register_checkpoint_conversion_mapping`
# and `WeightRenaming` (they were added alongside the new VLM loader), so we import
# defensively and skip registration on older/stock versions — those still honor the
# `_checkpoint_conversion_mapping` class attribute on the model, which we also set.
try:
    from transformers.conversion_mapping import (
        WeightRenaming,
        register_checkpoint_conversion_mapping,
    )

    register_checkpoint_conversion_mapping(
        PaliGemmaWMConfig.model_type,
        [
            WeightRenaming(source_patterns=r"^language_model.model", target_patterns="model.language_model"),
            WeightRenaming(source_patterns=r"^language_model.lm_head", target_patterns="lm_head"),
            WeightRenaming(source_patterns=r"^vision_tower", target_patterns="model.vision_tower"),
            WeightRenaming(source_patterns=r"^multi_modal_projector", target_patterns="model.multi_modal_projector"),
            WeightRenaming(source_patterns=r"^action_projector", target_patterns="model.action_projector"),
        ],
        overwrite=True,
    )
except ImportError:
    pass

__all__ = [
    "PaliGemmaWMConfig",
    "PaliGemmaWMForConditionalGeneration",
    "PaliGemmaWMModel",
    "PaliGemmaWMPreTrainedModel",
    "PaliGemmaWMProcessor",
    # "MultiHorizonSWMDataset",
]
