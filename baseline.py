"""One-command, bounded TinyStories baseline. Never saves source text."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import torch
import earnest

DATASET = 'https://huggingface.co/datasets/roneneldan/TinyStories'
REVISION = 'f54c09fd23315a6f9c86f9dc80f725de7d8f9c64'
PROMPTS = ['Once upon a time,', 'The little dog wanted to', 'Lily opened the door and']


def digest(story):
    return hashlib.sha256(b' '.join(story.split())).hexdigest()


def select_stories(raw, budget, exclude=(), min_bytes=257):
    # Only delimiter-terminated stories are accepted; discard the partial tail.
    stories, hashes, used = [], set(exclude), 0
    for part in raw.split(b'<|endoftext|>')[:-1]:
        story = part.strip() + b'\n'
        fingerprint = digest(story)
        if len(story) < min_bytes or fingerprint in hashes:
            continue
        story.decode('utf-8', errors='strict')
        if used + len(story) > budget:
            break
        stories.append(story)
        hashes.add(fingerprint)
        used += len(story)
    if not stories:
        raise ValueError('No complete eligible stories in the bounded response')
    return stories


def read_split(split, budget, revision, exclude=()):
    filename = {'train': 'TinyStories-train.txt', 'validation': 'TinyStories-valid.txt'}[split]
    url = f'{DATASET}/resolve/{revision}/{filename}'
    limit = budget + 65536
    request = Request(url, headers={'Range': f'bytes=0-{limit - 1}',
                                   'Accept-Encoding': 'identity', 'User-Agent': 'EarnestTrainer/0.2'})
    with urlopen(request, timeout=45) as response:
        if response.status == 206 and not response.headers.get('Content-Range', '').startswith('bytes 0-'):
            raise ValueError('Server returned an unexpected range')
        raw = response.read(limit)  # Also bounded if a server ignores Range.
    stories = select_stories(raw, budget, exclude)
    record = {'split': split, 'url': url, 'revision': revision, 'received_bytes': len(raw),
              'received_sha256': hashlib.sha256(raw).hexdigest(), 'selected_bytes': sum(map(len, stories)),
              'stories': len(stories), 'story_hashes': [digest(s) for s in stories],
              'selection': 'complete stories from bounded prefix; normalized exact duplicates excluded'}
    print(f"Read {split}: {len(stories)} stories, {record['selected_bytes']:,} bytes in RAM", flush=True)
    return stories, record


def tensorize(stories):
    return [torch.tensor(list(story), dtype=torch.long) for story in stories]


@torch.no_grad()
def validation_loss(model, documents, batches=32):
    model.eval()
    generator = random.Random(991)
    values = [earnest.loss_for(model, *earnest.batch(documents, model.cfg.context, 2, generator)).item()
              for _ in range(batches)]
    model.train()
    return sum(values) / len(values) / math.log(2)


@torch.no_grad()
def samples(model, seeds=(2026,)):
    model.eval()
    results = []
    # Sampling must not change the RNG state used by training/checkpoint resume.
    with torch.random.fork_rng():
        for seed in seeds:
            torch.manual_seed(seed)
            for prompt in PROMPTS:
                value = bytearray(prompt.encode())
                for _ in range(120):
                    tokens = torch.tensor([list(value[-model.cfg.context:])], dtype=torch.long)
                    probabilities = (model(tokens)[0, -1] / 0.8).softmax(-1)
                    value.append(torch.multinomial(probabilities, 1).item())
                text = value.decode('utf-8', errors='replace')
                text = ''.join(c if c.isprintable() or c in '\n\t' else '\ufffd' for c in text)
                results.append({'prompt': prompt, 'completion': text, 'seed': seed})
    model.train()
    return results


def benchmark(documents):
    measurements = []
    for context in (64, 256):
        torch.manual_seed(42)
        model = earnest.Earnest(earnest.Config(context=context))
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        rng = random.Random(42)
        durations = []
        for index in range(23):
            x, y = earnest.batch(documents, context, 2, rng)
            start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            loss = earnest.loss_for(model, x, y)
            loss.backward()
            opt.step()
            if index >= 3:
                durations.append(time.monotonic() - start)
        seconds = sum(durations) / len(durations)
        measurements.append({'context': context, 'seconds_per_step': seconds,
                             'bytes_per_second': context * 2 / seconds})
        print(f'Benchmark context {context}: {seconds:.4f}s/step', flush=True)
    return measurements


def append_json(path, value):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value) + '\n')


def write_report(folder, result, metadata):
    observations = [json.loads(line) for line in (folder / 'evaluations.jsonl').read_text().splitlines()]
    lines = ['# Earnest TinyStories baseline', '',
             f"Status: {result['status']}. Completed optimizer step: {result['step']}.", '',
             f"Random initialization; {metadata['parameters']:,} parameters; context {metadata['config']['context']} bytes; one CPU thread.", '',
             f"Training: {metadata['sources'][0]['stories']} complete stories ({metadata['sources'][0]['selected_bytes']:,} bytes).",
             f"Validation: {metadata['sources'][1]['stories']} separate complete stories ({metadata['sources'][1]['selected_bytes']:,} bytes).", '',
             f'Dataset: [TinyStories]({DATASET}), revision `{metadata["revision"]}`. License: CDLA-Sharing-1.0.', '',
             'Only bounded prefixes are read. No source corpus is saved. Story hashes and source metadata are retained. '
             'Train/validation exact duplicates are excluded after whitespace normalization; near duplicates are not detected.', '',
             '| Step | Validation bits/byte (lower is better) |', '|---:|---:|']
    lines += [f"| {row['step']} | {row['validation_bpb']:.4f} |" for row in observations]
    lines += ['', 'Validation uses the same 32 sampled batches at every checkpoint. It is a development signal, '
              'not an independent test-set result. Declining loss alone does not establish coherent language or reasoning.', '',
              '## Fixed-prompt samples', '']
    for row in (observations[0], observations[-1]):
        lines += [f"### Step {row['step']}", '']
        for sample in row['samples']:
            lines += ['```text', sample['completion'].replace('```', "'''"), '```', '']
    lines += ['## Continue', '', 'From this trainer directory:', '', '```powershell',
              f'.\\tinystories.ps1 --resume "{folder}" --steps 1000', '```', '',
              'Each invocation is capped at 1,000 updates and ten minutes of training by default. '
              '`--steps` is an additional update budget on resume. `latest.pt` stores the latest state; '
              '`best.pt` stores the lowest measured validation-loss checkpoint.']
    (folder / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    torch.set_num_threads(1)
    folder = Path(args.resume).resolve() if args.resume else Path(__file__).parent / 'runs' / datetime.now().strftime('tinystories-%Y%m%d-%H%M%S')
    if not args.resume:
        folder.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((folder / 'run.json').read_text()) if args.resume else None
    revision = metadata['revision'] if metadata else REVISION
    train_text, train_record = read_split('train', 1048576, revision)
    valid_text, valid_record = read_split('validation', 131072, revision, train_record['story_hashes'])
    records = [train_record, valid_record]
    if metadata and metadata['sources'] != records:
        raise ValueError('Source provenance mismatch; refusing to resume')
    training, validation = tensorize(train_text), tensorize(valid_text)
    del train_text, valid_text
    if not metadata:
        timings = benchmark(training)
        # Prefer longer context if the measured update cost is <= 0.5 seconds.
        context = 256 if timings[1]['seconds_per_step'] <= .5 else 64
        cfg = earnest.Config(context=context)
        metadata = {'revision': revision, 'config': asdict(cfg), 'sources': records,
                    'benchmarks': timings, 'seed': 42, 'threads': 1, 'batch_size': 2,
                    'torch': torch.__version__, 'evaluation_batches': 32,
                    'prompts': PROMPTS, 'sample_seed': 2026}
    cfg = earnest.Config(**metadata['config'])
    torch.manual_seed(42)
    model = earnest.Earnest(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = random.Random(42)
    step = 0
    if args.resume:
        state = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        generator.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        step = state['step']
        del state
    metadata['parameters'] = sum(p.numel() for p in model.parameters())
    (folder / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'Run: {folder}', flush=True)
    best = float('inf')
    if (folder / 'evaluations.jsonl').exists():
        best = min(json.loads(line)['validation_bpb'] for line in (folder / 'evaluations.jsonl').read_text().splitlines())

    def assess():
        nonlocal best
        value = validation_loss(model, validation)
        row = {'step': step, 'validation_bpb': value, 'samples': samples(model)}
        append_json(folder / 'evaluations.jsonl', row)
        earnest.save_checkpoint(folder / 'latest.pt', model, optimizer, step, generator, records)
        if value < best:
            best = value
            earnest.save_checkpoint(folder / 'best.pt', model, optimizer, step, generator, records)
        print(f'Step {step}: validation {value:.4f} bits/byte', flush=True)
        return value

    initial = assess()
    start = time.monotonic()
    initial_step = step
    last_assessed = step
    status = 'completed'
    try:
        for _ in range(args.steps):
            if time.monotonic() - start >= args.max_minutes * 60:
                status = 'time budget reached'
                break
            x, y = earnest.batch(training, cfg.context, 2, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = earnest.loss_for(model, x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                row = {'step': step, 'train_bpb': loss.item() / math.log(2), 'elapsed_seconds': time.monotonic() - start}
                append_json(folder / 'metrics.jsonl', row)
                print(json.dumps(row), flush=True)
                earnest.save_checkpoint(folder / 'latest.pt', model, optimizer, step, generator, records)
            if step % 200 == 0:
                assess()
                last_assessed = step
    except KeyboardInterrupt:
        status = 'interrupted'
    final = assess() if last_assessed != step else validation_loss(model, validation)
    result = {'status': status, 'step': step, 'updates_this_invocation': step - initial_step,
              'initial_validation_bpb': initial, 'final_validation_bpb': final,
              'seconds': time.monotonic() - start, 'folder': str(folder)}
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_report(folder, result, metadata)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', help='Existing baseline run directory')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--max-minutes', type=float, default=10)
    arguments = parser.parse_args()
    if arguments.steps < 1 or arguments.max_minutes <= 0:
        parser.error('Budgets must be positive')
    run(arguments)
