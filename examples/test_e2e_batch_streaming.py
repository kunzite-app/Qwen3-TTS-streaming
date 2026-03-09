import argparse
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF_AUDIO = Path(__file__).resolve().parents[1] / "kuklina-1.wav"
REF_TEXT = (
    "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? "
    "У него ведь куча наград по боевым искусствам."
)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(model_id: str, device: str) -> Qwen3TTSModel:
    dtype = pick_dtype(device)
    model = Qwen3TTSModel.from_pretrained(model_id, dtype=dtype)
    model.model.to(device)
    model.device = torch.device(device)
    return model


def collect_batch_stream(generator, batch_size: int) -> tuple[list[int], int]:
    per_item_chunks: list[list[np.ndarray]] = [[] for _ in range(batch_size)]
    sample_rate = None

    for chunks_list, sr in generator:
        if len(chunks_list) != batch_size:
            raise AssertionError(f"expected {batch_size} chunks, got {len(chunks_list)}")
        sample_rate = sr
        for idx, chunk in enumerate(chunks_list):
            if chunk.size > 0:
                per_item_chunks[idx].append(chunk)

    if sample_rate is None:
        raise AssertionError("batch stream did not emit audio")

    sample_counts = []
    for idx, chunks in enumerate(per_item_chunks):
        if not chunks:
            raise AssertionError(f"batch item {idx} emitted no audio")
        audio = np.concatenate(chunks)
        if audio.size == 0:
            raise AssertionError(f"batch item {idx} emitted empty audio")
        sample_counts.append(int(audio.size))

    return sample_counts, int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    model = load_model(BASE_MODEL, args.device)
    prompt_items = model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)

    texts = [
        "Hello from batch item one.",
        "This is the second batch streaming test item.",
        "Third item here, still short but enough to emit chunks.",
    ]

    sample_counts, sample_rate = collect_batch_stream(
        model.batch_stream_generate_voice_clone(
            text=texts,
            language="English",
            voice_clone_prompt=prompt_items,
            emit_every_frames=2,
            decode_window_frames=24,
            overlap_samples=128,
            max_frames=40,
            do_sample=False,
            subtalker_dosample=False,
            use_optimized_decode=False,
            first_chunk_emit_every=2,
            first_chunk_decode_window=24,
            first_chunk_frames=8,
        ),
        batch_size=len(texts),
    )

    counts_text = ",".join(str(v) for v in sample_counts)
    print(f"batch_stream_ok device={args.device} sr={sample_rate} samples={counts_text}")


if __name__ == "__main__":
    main()
