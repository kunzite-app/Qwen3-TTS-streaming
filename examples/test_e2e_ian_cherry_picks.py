import argparse
from pathlib import Path

import numpy as np
import torch

from qwen_tts import Qwen3TTSModel


BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
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


def collect_stream(generator) -> tuple[int, int, int]:
    chunks = []
    sample_rate = None
    for chunk, sr in generator:
        chunks.append(chunk)
        sample_rate = sr

    if not chunks or sample_rate is None:
        raise AssertionError("stream did not emit audio")

    audio = np.concatenate(chunks)
    if audio.size == 0:
        raise AssertionError("stream emitted empty audio")

    return len(chunks), int(audio.size), int(sample_rate)


def test_custom_voice(device: str) -> None:
    model = load_model(CUSTOM_MODEL, device)
    speakers = model.get_supported_speakers()
    if not speakers:
        raise AssertionError("custom voice model did not expose speakers")

    chunk_count, sample_count, sample_rate = collect_stream(
        model.stream_generate_custom_voice(
            text="Hello from the custom voice streaming smoke test.",
            speaker=speakers[0],
            language="English",
            emit_every_frames=4,
            decode_window_frames=40,
            overlap_samples=0,
            max_frames=48,
            do_sample=False,
            subtalker_dosample=False,
            use_optimized_decode=False,
            first_chunk_emit_every=2,
            first_chunk_decode_window=24,
            first_chunk_frames=8,
        )
    )
    print(
        f"custom_voice_stream_ok device={device} speaker={speakers[0]} "
        f"chunks={chunk_count} samples={sample_count} sr={sample_rate}"
    )


def test_base_overlap(device: str) -> None:
    model = load_model(BASE_MODEL, device)
    prompt_items = model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)

    chunk_count, sample_count, sample_rate = collect_stream(
        model.stream_generate_voice_clone(
            text="Это короткий тест потоковой озвучки с overlap.",
            language="Russian",
            voice_clone_prompt=prompt_items,
            emit_every_frames=4,
            decode_window_frames=40,
            overlap_samples=256,
            max_frames=48,
            do_sample=False,
            subtalker_dosample=False,
            use_optimized_decode=False,
            first_chunk_emit_every=2,
            first_chunk_decode_window=24,
            first_chunk_frames=8,
        )
    )
    print(
        f"base_overlap_stream_ok device={device} chunks={chunk_count} "
        f"samples={sample_count} sr={sample_rate}"
    )


def test_base_eos(device: str) -> None:
    model = load_model(BASE_MODEL, device)
    prompt_items = model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)

    chunk_count, sample_count, sample_rate = collect_stream(
        model.stream_generate_voice_clone(
            text="Hello.",
            language="English",
            voice_clone_prompt=prompt_items,
            emit_every_frames=2,
            decode_window_frames=24,
            overlap_samples=0,
            max_frames=32,
            do_sample=False,
            subtalker_dosample=False,
            use_optimized_decode=False,
        )
    )
    print(
        f"base_eos_stream_ok device={device} chunks={chunk_count} "
        f"samples={sample_count} sr={sample_rate}"
    )


def test_zero_overlap_flush(device: str) -> None:
    model = load_model(BASE_MODEL, device)
    prompt_items = model.create_voice_clone_prompt(ref_audio=str(REF_AUDIO), ref_text=REF_TEXT)

    chunk_count, sample_count, sample_rate = collect_stream(
        model.stream_generate_voice_clone(
            text="Tiny zero-overlap flush test.",
            language="English",
            voice_clone_prompt=prompt_items,
            emit_every_frames=2,
            decode_window_frames=24,
            overlap_samples=0,
            max_frames=24,
            do_sample=False,
            subtalker_dosample=False,
            use_optimized_decode=False,
            first_chunk_emit_every=2,
            first_chunk_decode_window=24,
            first_chunk_frames=8,
        )
    )
    print(
        f"zero_overlap_flush_ok device={device} chunks={chunk_count} "
        f"samples={sample_count} sr={sample_rate}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=pick_device())
    args = parser.parse_args()

    print(f"using_device={args.device}")
    test_custom_voice(args.device)
    test_base_overlap(args.device)
    test_base_eos(args.device)
    test_zero_overlap_flush(args.device)


if __name__ == "__main__":
    main()
