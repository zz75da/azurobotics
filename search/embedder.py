"""all-MiniLM-L6-v2 sentence embeddings with ONNX Runtime, without PyTorch.

Reproduces what sentence-transformers does for this model: tokenize (max 256 tokens), run the
transformer, mean-pool the token embeddings with the attention mask, then L2-normalise. The ONNX
export and tokenizer come from the official model repository at a pinned revision, and the ONNX
file is checked against its published SHA-256. tests/test_embedder_parity.py proves the vectors
match sentence-transformers.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence

import numpy as np

MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
ONNX_SHA256 = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
MAX_TOKENS = 256
DIMENSIONS = 384


def download_model(target: Path) -> Path:
    """Fetch model.onnx and tokenizer.json at the pinned revision and verify the ONNX file."""
    from huggingface_hub import hf_hub_download

    target.mkdir(parents=True, exist_ok=True)
    onnx_path = Path(hf_hub_download(MODEL_REPO, "onnx/model.onnx", revision=MODEL_REVISION, local_dir=target))
    hf_hub_download(MODEL_REPO, "tokenizer.json", revision=MODEL_REVISION, local_dir=target)
    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    if digest != ONNX_SHA256:
        raise RuntimeError("model.onnx checksum mismatch: {}".format(digest))
    return target


class MiniLMEmbedder:
    def __init__(self, model_dir: Path):
        import onnxruntime
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        self.session = onnxruntime.InferenceSession(
            str(model_dir / "onnx" / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        encodings = self.tokenizer.encode_batch(list(texts))
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        token_embeddings = self.session.run(None, feeds)[0]  # (batch, tokens, 384)
        weights = mask[..., None].astype(np.float32)
        pooled = (token_embeddings * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        pooled /= np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        return pooled.astype(np.float32).tolist()
