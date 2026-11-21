"""Sentence embeddings with all-MiniLM-L6-v2, run through ONNX Runtime (no PyTorch).

The model is the official ONNX export published by the sentence-transformers team. We reproduce its
pipeline exactly: WordPiece tokenizer -> transformer -> mean pooling over real tokens -> L2 normalisation
(the model's modules.json: Transformer, Pooling(mean), Normalize). Files are pinned to a repository
revision and verified by SHA-256, like the datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from recsys.data.download import _stream_to_file, verify
from recsys.paths import models_dir


@dataclass(frozen=True)
class TextModelSpec:
    name: str
    repo: str
    revision: str
    files: dict[str, str] = field(default_factory=dict)  # remote path -> SHA-256
    dim: int = 384

    def url(self, remote: str) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{remote}"


MINILM = TextModelSpec(
    name="all-MiniLM-L6-v2",
    repo="sentence-transformers/all-MiniLM-L6-v2",
    revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    files={
        "onnx/model.onnx": "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
        "tokenizer.json": "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
    },
)


def fetch_text_model(spec: TextModelSpec = MINILM, dest: Path | None = None) -> Path:
    """Download (once) and verify the encoder files; returns the folder holding model.onnx and
    tokenizer.json."""
    dest = dest or models_dir("text", spec.name)
    dest.mkdir(parents=True, exist_ok=True)
    for remote, sha in spec.files.items():
        local = dest / Path(remote).name
        if not local.exists():
            _stream_to_file(spec.url(remote), local)
        verify(local, sha256=sha)
    return dest


class TextEncoder:
    """Encode texts into unit-length 384-d vectors (cosine similarity = dot product)."""

    def __init__(self, model_dir: Path, max_length: int = 128, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer.encode_batch(texts[start : start + batch_size])
            ids = np.array([e.ids for e in batch], dtype=np.int64)
            mask = np.array([e.attention_mask for e in batch], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self._inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            tokens = self.session.run(None, feeds)[0]  # (batch, seq, dim): last hidden state
            m = mask[..., None].astype(np.float32)
            pooled = (tokens * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
            out.append(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None))
        if not out:
            return np.zeros((0, MINILM.dim), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)
