from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_causal_memory_lm import _record_to_document, render_document_for_hf_tokenizer  # noqa: E402

SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>", "<|im_start|>", "<|im_end|>"]


def iter_training_text(path: Path, *, max_docs: int | None = None):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip().lstrip("\ufeff")
            if not line:
                continue
            document = _record_to_document(json.loads(line), source=f"{path}:{line_number}")
            if document is None:
                continue
            text = render_document_for_hf_tokenizer(document)
            if text:
                yield text
                count += 1
                if max_docs is not None and count >= max_docs:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local Hugging Face fast ByteLevel BPE tokenizer.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-docs", type=int)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )
    tokenizer.train_from_iterator(
        iter_training_text(input_path, max_docs=args.max_docs),
        trainer=trainer,
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )
    fast.save_pretrained(str(output_dir))
    meta = {
        "input": str(input_path),
        "vocab_size_requested": args.vocab_size,
        "vocab_size_actual": len(fast),
        "min_frequency": args.min_frequency,
        "max_docs": args.max_docs,
        "special_tokens": SPECIAL_TOKENS,
    }
    (output_dir / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
