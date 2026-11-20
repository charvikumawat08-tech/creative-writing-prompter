"""Command-line entry point: `python -m app.cli --help`."""
import argparse
import math
from pathlib import Path

from app.core import NGramModel, tokenize
from app.storage import load_corpus


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="creative-writing-prompter",
        description="Get context-aware next-word suggestions from a short-story corpus.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    suggest_parser = sub.add_parser("suggest", help="suggest or generate text from a prompt")
    suggest_parser.add_argument("prompt", nargs="+", help="seed text for the model")
    suggest_parser.add_argument(
        "--corpus", type=Path, help="UTF-8 corpus file (defaults to the bundled stories)"
    )
    suggest_parser.add_argument("--n", type=_positive_int, default=2, help="n-gram order")
    suggest_parser.add_argument("--k", type=_positive_float, default=1.0, help="Add-k smoothing")
    suggest_parser.add_argument(
        "--top-n", type=_positive_int, default=5, help="number of suggestions to show"
    )
    suggest_parser.add_argument(
        "--length", type=_non_negative_int, default=0, help="number of words to generate"
    )
    suggest_parser.add_argument(
        "--seed", type=int, help="optional random seed for reproducible generation"
    )

    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Creative Writing Prompter is ready. Open README.md for your milestones.")
        return

    prompt = " ".join(args.prompt)
    try:
        model = NGramModel(
            tokenize(load_corpus(args.corpus)), n=args.n, k=args.k
        )
    except OSError as exc:
        parser.error(f"could not read corpus: {exc}")
    except ValueError as exc:
        parser.error(str(exc))

    print(f"Suggestions for {prompt!r}:")
    for word, probability in model.suggest_next_words(prompt, top_n=args.top_n):
        print(f"  {word}: {probability:.4f}")
    if args.length:
        print(
            "Generated text: "
            + model.generate_text(prompt, length=args.length, seed=args.seed)
        )


if __name__ == "__main__":
    main()
