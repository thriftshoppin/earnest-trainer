# Earnest Trainer

Earnest is an experimental byte-level language-model trainer for modest hardware. It runs on CPU with one thread by default. Its stacked trainer grows a model one block at a time, freezes accepted blocks, and can cache frozen activations on disk. The desktop monitor shows training metrics, checkpoints, and generated samples.

## Windows setup

Install 64-bit Python with Tk support. The dependency lock reflects the CPU build used with Python 3.14. In the repository root, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
.\.venv\Scripts\python.exe -m unittest -v
```

## Train with local text

Supply separate UTF-8 text files for training and validation. The example uses a 256-byte context and the 256-value byte vocabulary.

```powershell
.\stacked.ps1 new --stack stacks/local-crawl --context 256 --local-train train.txt --local-validation validation.txt
.\show-trainer.ps1 --stack stacks/local-crawl
```

For command-line training and sample review, use:

```powershell
.\stacked.ps1 train --stack stacks/local-crawl --steps 1000
.\stacked.ps1 crawl-review --stack stacks/local-crawl
```

The crawl review displays raw continuations and held-out bits per byte. The monitor lets you inspect outputs before freezing a crawl block. Run `./stacked.ps1 --help` for more commands. Checkpoints and activation caches are written to the stack directory.

For a quick trainer smoke run using built-in synthetic text, use `./earnest.ps1 train --smoke --steps 100 --out runs/smoke`. The optional TinyStories baseline is available through `./tinystories.ps1`.

## Rights

Earnest trainer source and documentation are all rights reserved under [LICENSE](LICENSE). The bundled CMU Pronouncing Dictionary has its own [license](lexicon/LICENSE).
