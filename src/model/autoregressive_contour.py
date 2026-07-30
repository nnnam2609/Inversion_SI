"""One-frame-reference autoregressive contour inversion.

The audio encoder intentionally mirrors the dense/Bi-LSTM frontend in
``baseline_5.BaselineModel``.  It lives in a separate model so the historical
baseline path and checkpoint format remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mlp(input_size: int, hidden_size: int, output_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.ReLU(),
    )


class BaselineAudioEncoder(nn.Module):
    """Numerically compatible copy of the baseline ST5 audio frontend."""

    checkpoint_prefixes = ("ff_layer_1", "ff_layer_2", "lstm_layer_1", "lstm_layer_2")

    def __init__(self, input_dimension: int, hidden_dimension: int, num_layers: int = 1):
        super().__init__()
        self.input_dimension = int(input_dimension)
        self.hidden_dimension = int(hidden_dimension)
        self.num_layers = int(num_layers)
        self.output_dimension = 2 * self.hidden_dimension
        self.ff_layer_1 = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.ff_layer_2 = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.lstm_layer_1 = nn.LSTM(
            self.hidden_dimension,
            self.hidden_dimension,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_layer_2 = nn.LSTM(
            2 * self.hidden_dimension,
            self.hidden_dimension,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"audio must be [batch,time,features], got {tuple(audio.shape)}")
        time_steps = int(audio.shape[1])
        cpu_lengths = lengths.detach().to(device="cpu", dtype=torch.long).clamp(1, time_steps)
        hidden = torch.relu(self.ff_layer_1(audio))
        hidden = torch.relu(self.ff_layer_2(hidden))
        packed = pack_padded_sequence(hidden, cpu_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm_layer_1(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=time_steps)
        hidden = torch.relu(hidden)
        packed = pack_padded_sequence(hidden, cpu_lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm_layer_2(packed)
        hidden, _ = pad_packed_sequence(packed, batch_first=True, total_length=time_steps)
        return torch.relu(hidden)

    def load_baseline_checkpoint(self, checkpoint_path: str) -> dict[str, list[str]]:
        """Load only matching frontend parameters and fail on any mismatch."""
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload
        if isinstance(payload, dict):
            for key in ("model_state_dict", "state_dict"):
                if isinstance(payload.get(key), dict):
                    state = payload[key]
                    break
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported baseline checkpoint payload: {type(state)!r}")

        own_state = self.state_dict()
        selected: dict[str, torch.Tensor] = {}
        unexpected: list[str] = []
        shape_mismatches: list[str] = []
        for raw_key, value in state.items():
            key = str(raw_key)
            for prefix in ("module.", "model.", "audio_encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
            if not key.startswith(self.checkpoint_prefixes):
                continue
            if key not in own_state:
                unexpected.append(key)
            elif tuple(value.shape) != tuple(own_state[key].shape):
                shape_mismatches.append(
                    f"{key}: checkpoint={tuple(value.shape)} model={tuple(own_state[key].shape)}"
                )
            else:
                selected[key] = value
        missing = sorted(set(own_state) - set(selected))
        if missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "Audio checkpoint loading failed: "
                f"missing={missing}, unexpected={unexpected}, shape_mismatches={shape_mismatches}"
            )
        self.load_state_dict(selected, strict=True)
        return {"loaded": sorted(selected), "missing": [], "unexpected": []}


@dataclass(frozen=True)
class AutoregressiveDiagnostics:
    anchor_embedding_norm: float
    history_embedding_norm: float
    audio_embedding_norm: float
    decoder_state_norm: float


class AnchoredAutoregressiveContour(nn.Module):
    """Anchor-relative free-running contour decoder.

    Frame zero is copied exactly from ``anchor``.  All later predictions are
    absolute residuals relative to that anchor, which prevents integration
    drift.  ``infer`` accepts no post-anchor target tensor by construction.
    """

    def __init__(
        self,
        input_dimension: int,
        audio_hidden_dimension: int,
        num_layers: int,
        num_articulators: int,
        output_layer: int,
        *,
        embedding_hidden_dimension: int = 256,
        embedding_dimension: int = 128,
        decoder_hidden_dimension: int = 512,
        output_hidden_dimension: int = 512,
        balance_decoder_embeddings: bool = False,
    ):
        super().__init__()
        self.num_articulators = int(num_articulators)
        self.output_layer = int(output_layer)
        self.contour_dimension = self.num_articulators * self.output_layer
        self.decoder_hidden_dimension = int(decoder_hidden_dimension)
        self.balance_decoder_embeddings = bool(balance_decoder_embeddings)
        self.audio_encoder = BaselineAudioEncoder(
            input_dimension, audio_hidden_dimension, num_layers=num_layers
        )
        audio_dimension = self.audio_encoder.output_dimension
        self.anchor_encoder = _mlp(
            self.contour_dimension, embedding_hidden_dimension, embedding_dimension
        )
        self.history_encoder = _mlp(
            self.contour_dimension, embedding_hidden_dimension, embedding_dimension
        )
        self.decoder_initializer = nn.Sequential(
            nn.Linear(audio_dimension + embedding_dimension, decoder_hidden_dimension),
            nn.Tanh(),
        )
        self.decoder_cell = nn.GRUCell(
            audio_dimension + 2 * embedding_dimension, decoder_hidden_dimension
        )
        self.output_head = nn.Sequential(
            nn.Linear(decoder_hidden_dimension, output_hidden_dimension),
            nn.ReLU(),
            nn.Linear(output_hidden_dimension, self.contour_dimension),
        )

    @classmethod
    def from_config(cls, config: dict) -> "AnchoredAutoregressiveContour":
        return cls(
            input_dimension=int(config["input_layer"]),
            audio_hidden_dimension=int(config["hidden_layer"]),
            num_layers=int(config.get("num_layers", 1)),
            num_articulators=len(config["classes"]),
            output_layer=int(config["output_layer"]),
            embedding_hidden_dimension=int(config.get("embedding_hidden_dimension", 256)),
            embedding_dimension=int(config.get("embedding_dimension", 128)),
            decoder_hidden_dimension=int(config.get("decoder_hidden_dimension", 512)),
            output_hidden_dimension=int(config.get("output_hidden_dimension", 512)),
            balance_decoder_embeddings=bool(
                config.get("balance_decoder_embeddings", False)
            ),
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        mask = steps < lengths.to(device=hidden.device, dtype=torch.long).unsqueeze(1)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1).to(hidden.dtype)
        return (hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator

    def _validate_inputs(
        self,
        audio: torch.Tensor,
        anchor: torch.Tensor,
        lengths: torch.Tensor,
        targets: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if anchor.ndim == 3:
            anchor_flat = anchor.reshape(anchor.shape[0], -1)
        elif anchor.ndim == 2:
            anchor_flat = anchor
        else:
            raise ValueError(f"anchor must be [B,A,P] or [B,D], got {tuple(anchor.shape)}")
        if anchor_flat.shape != (audio.shape[0], self.contour_dimension):
            raise ValueError(
                f"anchor shape resolves to {tuple(anchor_flat.shape)}, expected "
                f"({audio.shape[0]},{self.contour_dimension})"
            )
        if lengths.numel() != audio.shape[0]:
            raise ValueError("one sequence length is required per batch item")
        targets_flat = None
        if targets is not None:
            if targets.ndim == 4:
                targets_flat = targets.reshape(targets.shape[0], targets.shape[1], -1)
            elif targets.ndim == 3:
                targets_flat = targets
            else:
                raise ValueError("targets must be [B,T,A,P] or [B,T,D]")
            if targets_flat.shape[:2] != audio.shape[:2]:
                raise ValueError("target batch/time dimensions must match audio")
            if targets_flat.shape[-1] != self.contour_dimension:
                raise ValueError("target contour dimension does not match configuration")
        return anchor_flat, targets_flat

    def forward(
        self,
        audio: torch.Tensor,
        lengths: torch.Tensor,
        anchor: torch.Tensor,
        *,
        targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        sampling_mode: Literal["per_sequence", "per_step"] = "per_sequence",
        generator: torch.Generator | None = None,
        history_mode: Literal["normal", "clamp", "shuffle"] = "normal",
        reset_decoder_state: bool = False,
        shuffle_audio_time: bool = False,
        collect_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, AutoregressiveDiagnostics]:
        anchor_flat, targets_flat = self._validate_inputs(audio, anchor, lengths, targets)
        ratio = float(teacher_forcing_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must be in [0,1]")
        if ratio > 0.0 and targets_flat is None:
            raise ValueError("teacher forcing requires targets")

        audio_for_encoder = audio
        if shuffle_audio_time and audio.shape[1] > 1:
            permutation = torch.randperm(
                audio.shape[1], device=audio.device, generator=generator
            )
            audio_for_encoder = audio[:, permutation]
        encoded_audio = self.audio_encoder(audio_for_encoder, lengths)
        anchor_embedding = self.anchor_encoder(anchor_flat)
        if self.balance_decoder_embeddings:
            encoded_audio = torch.nn.functional.layer_norm(
                encoded_audio,
                (encoded_audio.shape[-1],),
            )
            anchor_embedding = torch.nn.functional.layer_norm(
                anchor_embedding,
                (anchor_embedding.shape[-1],),
            )
        pooled_audio = self._masked_mean(encoded_audio, lengths)
        initial_state = self.decoder_initializer(torch.cat((anchor_embedding, pooled_audio), dim=-1))
        state = initial_state
        predictions = [anchor_flat]
        history_norms: list[torch.Tensor] = []
        state_norms: list[torch.Tensor] = []

        sequence_teacher_mask = None
        if ratio not in (0.0, 1.0) and sampling_mode == "per_sequence":
            sequence_teacher_mask = torch.rand(
                (audio.shape[0], 1), device=audio.device, generator=generator
            ) < ratio

        for step in range(1, audio.shape[1]):
            previous = predictions[-1]
            if ratio == 1.0:
                previous = targets_flat[:, step - 1]
            elif ratio > 0.0:
                if sampling_mode == "per_step":
                    teacher_mask = torch.rand(
                        (audio.shape[0], 1), device=audio.device, generator=generator
                    ) < ratio
                else:
                    teacher_mask = sequence_teacher_mask
                previous = torch.where(teacher_mask, targets_flat[:, step - 1], previous)

            if history_mode == "clamp":
                previous = anchor_flat
            elif history_mode == "shuffle":
                previous = previous.roll(1, dims=0)
            elif history_mode != "normal":
                raise ValueError(f"Unknown history mode: {history_mode!r}")

            history_embedding = self.history_encoder(previous - anchor_flat)
            if self.balance_decoder_embeddings:
                history_embedding = torch.nn.functional.layer_norm(
                    history_embedding,
                    (history_embedding.shape[-1],),
                )
            if collect_diagnostics:
                history_norms.append(history_embedding.norm(dim=-1).mean())
            decoder_input = torch.cat(
                (encoded_audio[:, step], anchor_embedding, history_embedding), dim=-1
            )
            proposed_state = self.decoder_cell(
                decoder_input, initial_state if reset_decoder_state else state
            )
            active = (step < lengths.to(device=audio.device)).unsqueeze(1)
            state = torch.where(active, proposed_state, state)
            if collect_diagnostics:
                state_norms.append(state.norm(dim=-1).mean())
            residual = self.output_head(state)
            prediction = anchor_flat + residual
            prediction = torch.where(active, prediction, anchor_flat)
            predictions.append(prediction)

        output = torch.stack(predictions, dim=1).reshape(
            audio.shape[0],
            audio.shape[1],
            self.num_articulators,
            self.output_layer,
        )
        zero = output.new_tensor(0.0)
        if collect_diagnostics:
            diagnostics = AutoregressiveDiagnostics(
                anchor_embedding_norm=float(anchor_embedding.detach().norm(dim=-1).mean().cpu()),
                history_embedding_norm=float(
                    torch.stack(history_norms).detach().mean().cpu() if history_norms else zero.cpu()
                ),
                audio_embedding_norm=float(encoded_audio.detach().norm(dim=-1).mean().cpu()),
                decoder_state_norm=float(
                    torch.stack(state_norms).detach().mean().cpu() if state_norms else zero.cpu()
                ),
            )
        else:
            diagnostics = AutoregressiveDiagnostics(0.0, 0.0, 0.0, 0.0)
        return output, diagnostics

    @torch.no_grad()
    def infer(
        self,
        audio: torch.Tensor,
        anchor: torch.Tensor,
        lengths: torch.Tensor,
        *,
        history_mode: Literal["normal", "clamp", "shuffle"] = "normal",
        reset_decoder_state: bool = False,
        shuffle_audio_time: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, AutoregressiveDiagnostics]:
        was_training = self.training
        self.eval()
        try:
            return self(
                audio,
                lengths,
                anchor,
                teacher_forcing_ratio=0.0,
                history_mode=history_mode,
                reset_decoder_state=reset_decoder_state,
                shuffle_audio_time=shuffle_audio_time,
                generator=generator,
            )
        finally:
            self.train(was_training)


class BaselineAudioOnly(nn.Module):
    """Small evaluation-only wrapper for historical baseline checkpoints."""

    def __init__(self, config: dict):
        super().__init__()
        self.num_articulators = len(config["classes"])
        self.output_layer = int(config["output_layer"])
        self.audio_encoder = BaselineAudioEncoder(
            int(config["input_layer"]),
            int(config["hidden_layer"]),
            int(config.get("num_layers", 1)),
        )
        self.readout_layer = nn.Linear(
            self.audio_encoder.output_dimension,
            self.num_articulators * self.output_layer,
        )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise RuntimeError("Unsupported baseline checkpoint payload")
        encoder_state = {}
        for key in self.audio_encoder.state_dict():
            if key not in state:
                raise RuntimeError(f"Baseline checkpoint is missing {key}")
            encoder_state[key] = state[key]
        self.audio_encoder.load_state_dict(encoder_state, strict=True)
        readout = {
            key.removeprefix("readout_layer."): value
            for key, value in state.items()
            if key.startswith("readout_layer.")
        }
        self.readout_layer.load_state_dict(readout, strict=True)

    def forward(self, audio: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        hidden = self.audio_encoder(audio, lengths)
        flat = self.readout_layer(hidden)
        return flat.reshape(
            audio.shape[0], audio.shape[1], self.num_articulators, self.output_layer
        )
