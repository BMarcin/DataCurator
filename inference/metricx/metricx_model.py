# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MetricX model + batched scorer for the inference service.

``MT5ForRegression`` is copied verbatim (Apache-2.0) from the upstream MetricX
repo (google-research/metricx, ``metricx23/models.py``) because it subclasses
private ``transformers.models.mt5.modeling_mt5`` internals and reads the score
from a single vocab logit — a stock HF class cannot stand in for it. The
upstream ``vendor/`` tree is gitignored in this repo, so the class is committed
here to keep the service self-contained (no DataCurator or vendor dependency).

``MetricXScorer`` wraps the model with a *batched* scorer, reproducing
``metricx23/predict.py`` / ``metricx24/predict.py`` exactly (input templates —
see ``variants.py`` — EOS removal, 0-25 clamp) but padding a whole batch in one
forward pass instead of the upstream's one-example-at-a-time path (upstream has
no padding collator, so it can only run at batch_size 1). The same class serves
both checkpoints: the upstream 23 and 24 model files are byte-identical.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import warnings
from typing import Dict, List, Optional, Tuple, Union

import torch
import transformers
import transformers.modeling_outputs
from loguru import logger
from torch import nn

import variants

BaseModelOutput = transformers.modeling_outputs.BaseModelOutput
ModelOutput = transformers.modeling_outputs.ModelOutput

MT5Config = transformers.models.mt5.modeling_mt5.MT5Config
MT5PreTrainedModel = transformers.models.mt5.modeling_mt5.MT5PreTrainedModel
MT5Stack = transformers.models.mt5.modeling_mt5.MT5Stack

__HEAD_MASK_WARNING_MSG = (
    transformers.models.mt5.modeling_mt5.__HEAD_MASK_WARNING_MSG  # pylint: disable=protected-access
)


@dataclasses.dataclass
class MT5ForRegressionOutput(ModelOutput):
  loss: Optional[torch.FloatTensor] = None
  predictions: torch.FloatTensor = None


class MT5ForRegression(MT5PreTrainedModel):
  """MT5 model for regression."""

  def __init__(self, config: MT5Config):
    super().__init__(config)
    self.model_dim = config.d_model

    self.shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    self.encoder = MT5Stack(encoder_config, self.shared)

    decoder_config = copy.deepcopy(config)
    decoder_config.is_decoder = True
    decoder_config.is_encoder_decoder = False
    decoder_config.num_layers = config.num_decoder_layers
    self.decoder = MT5Stack(decoder_config, self.shared)

    self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    # Initialize weights and apply final processing
    self.post_init()

    # Model parallel
    self.model_parallel = False
    self.device_map = None

  def forward(
      self,
      input_ids: Optional[torch.LongTensor] = None,
      attention_mask: Optional[torch.FloatTensor] = None,
      decoder_attention_mask: Optional[torch.BoolTensor] = None,
      head_mask: Optional[torch.FloatTensor] = None,
      decoder_head_mask: Optional[torch.FloatTensor] = None,
      cross_attn_head_mask: Optional[torch.Tensor] = None,
      encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
      inputs_embeds: Optional[torch.FloatTensor] = None,
      decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
      labels: Optional[torch.FloatTensor] = None,
      use_cache: Optional[bool] = None,
      output_attentions: Optional[bool] = None,
      output_hidden_states: Optional[bool] = None,
      return_dict: Optional[bool] = None,
  ) -> Union[Tuple[torch.FloatTensor], MT5ForRegressionOutput]:
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # FutureWarning: head_mask was separated into two input args - head_mask,
    # decoder_head_mask
    if head_mask is not None and decoder_head_mask is None:
      if self.config.num_layers == self.config.num_decoder_layers:
        warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
        decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
      # Convert encoder inputs in embeddings if needed
      encoder_outputs = self.encoder(
          input_ids=input_ids,
          attention_mask=attention_mask,
          inputs_embeds=inputs_embeds,
          head_mask=head_mask,
          output_attentions=output_attentions,
          output_hidden_states=output_hidden_states,
          return_dict=return_dict,
      )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
      encoder_outputs = BaseModelOutput(
          last_hidden_state=encoder_outputs[0],
          hidden_states=encoder_outputs[1]
          if len(encoder_outputs) > 1
          else None,
          attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
      )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)

    # Create 1 step of dummy input for the decoder.
    batch_size = input_ids.size(0)
    decoder_input_ids = torch.LongTensor([0]).repeat(batch_size).reshape(-1, 1)
    if torch.cuda.is_available():
      decoder_input_ids = decoder_input_ids.to(torch.device("cuda"))

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.decoder.first_device)
      hidden_states = hidden_states.to(self.decoder.first_device)
      if decoder_input_ids is not None:
        decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
      if attention_mask is not None:
        attention_mask = attention_mask.to(self.decoder.first_device)
      if decoder_attention_mask is not None:
        decoder_attention_mask = decoder_attention_mask.to(
            self.decoder.first_device
        )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
      torch.cuda.set_device(self.encoder.first_device)
      self.lm_head = self.lm_head.to(self.encoder.first_device)
      sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
      # Rescale output before projecting on vocab
      # See
      # https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
      sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)

    # 250089 = <extra_id_10>
    predictions = lm_logits[:, 0, 250089]

    # Clip to 0 to 25
    predictions = torch.clamp(predictions, 0, 25)

    loss = None
    if labels is not None:
      loss_fct = nn.MSELoss()
      # move labels to correct device to enable PP
      labels = labels.to(predictions.device)
      loss = loss_fct(predictions.view(-1), labels.view(-1))

    return MT5ForRegressionOutput(
        loss=loss,
        predictions=predictions,
    )


# --------------------------------------------------------------------------- #
# Batched QE scorer (service-side, not from upstream)
# --------------------------------------------------------------------------- #

_DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


class MetricXScorer:
    """Loads MetricX once and scores (source, hypothesis[, reference]) in batches.

    Reproduces upstream ``predict.py`` exactly for the configured variant: the
    input template comes from ``variants.make_input``, the trailing mT5 EOS
    token is dropped, and the score is the ``<extra_id_10>`` logit clamped to
    [0, 25] (lower is better). The only departure is that a batch is padded and
    run in one forward pass — correct because the single decoder step is length
    independent and encoder padding is masked out by ``attention_mask``.
    """

    def __init__(
        self,
        model: MT5ForRegression,
        tokenizer,
        *,
        device: torch.device,
        max_input_length: int,
        variant: str = variants.METRICX_23_QE,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_input_length = max_input_length
        self.variant = variants.check_variant(variant)
        self.accepts_reference = variants.accepts_reference(variant)

    @classmethod
    def load(
        cls,
        model_name: str,
        tokenizer_name: str,
        *,
        device: Optional[str] = None,
        dtype: str = "float32",
        max_input_length: int = 1024,
        variant: str = variants.METRICX_23_QE,
    ) -> "MetricXScorer":
        torch_device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        torch_dtype = _DTYPES.get(dtype.lower())
        if torch_dtype is None:
            raise ValueError(f"unknown METRICX_DTYPE {dtype!r}; use one of {sorted(_DTYPES)}")

        logger.info("loading tokenizer {}...", tokenizer_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info("loading weights {} (downloads on first run)...", model_name)
        # Materialise the weights in the target dtype up front. Loading fp32 and
        # casting after the .to(device) would spike peak VRAM to the fp32
        # footprint (2x for bf16/fp16) even though the steady state is smaller.
        model = MT5ForRegression.from_pretrained(model_name, torch_dtype=torch_dtype)
        logger.info("moving model to {} ({})...", torch_device, torch_dtype)
        model.to(torch_device)
        model.eval()
        return cls(
            model,
            tokenizer,
            device=torch_device,
            max_input_length=max_input_length,
            variant=variant,
        )

    def _encode(self, pair: Dict[str, str]) -> Dict[str, list]:
        text = variants.make_input(
            self.variant, pair["source"], pair["hypothesis"], pair.get("reference")
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_input_length,
            truncation=True,
            padding=False,
        )
        # MetricX drops the trailing mT5 EOS token (predict.py `_remove_eos`).
        enc["input_ids"] = enc["input_ids"][:-1]
        enc["attention_mask"] = enc["attention_mask"][:-1]
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    @torch.inference_mode()
    def score_batch(self, pairs: List[Dict[str, str]]) -> List[float]:
        """Score ``{"source", "hypothesis"[, "reference"]}`` dicts (0-25, lower better)."""
        if not pairs:
            return []
        encs = [self._encode(p) for p in pairs]
        batch = self.tokenizer.pad(encs, return_tensors="pt").to(self.device)
        preds = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).predictions
        return preds.float().cpu().tolist()


def load_scorer_from_env() -> MetricXScorer:
    """Build a scorer from ``METRICX_*`` environment variables.

    Used as the server's ``model_loader``; runs on a background thread at
    startup so the HTTP surface can come up while the ~15GB weights load.
    """
    model_name = os.environ.get("METRICX_MODEL", "google/metricx-23-qe-xl-v2p0")
    # validate before load() kicks off the ~15GB weight download
    variant = variants.check_variant(
        os.environ.get("METRICX_VARIANT") or variants.detect_variant(model_name)
    )
    return MetricXScorer.load(
        model_name,
        os.environ.get("METRICX_TOKENIZER", "google/mt5-xl"),
        device=os.environ.get("METRICX_DEVICE") or None,
        dtype=os.environ.get("METRICX_DTYPE", "float32"),
        max_input_length=int(
            os.environ.get("METRICX_MAX_INPUT_LENGTH")
            or variants.default_max_input_length(variant)
        ),
        variant=variant,
    )
