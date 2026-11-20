import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_suggest_command_supports_options_and_custom_corpus(tmp_path):
    corpus = tmp_path / "stories.txt"
    corpus.write_text(
        "the moon glows the moon dims",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.cli",
            "suggest",
            "the",
            "moon",
            "--corpus",
            str(corpus),
            "--k",
            "0.5",
            "--top-n",
            "2",
            "--length",
            "3",
            "--seed",
            "12",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Suggestions for 'the moon':" in result.stdout
    assert "Generated text:" in result.stdout
    assert len(result.stdout.split("Generated text: ", 1)[1].split()) == 5


def test_suggest_rejects_non_positive_smoothing():
    result = subprocess.run(
        [sys.executable, "-m", "app.cli", "suggest", "once", "--k", "0"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "must be a finite number greater than 0" in result.stderr
