"""Stack-by-stack trainer for Earnest: every stage trains ONE new block on top of the frozen earlier ones.

A stack is a folder. Each finished block is its own file there, so the model grows on disk one run at a time:

    stack.json        width / heads / context / stride, data revision, one entry per frozen block
    embed.pt          byte + position embeddings (trained in stage 0, frozen afterwards)
    block-000.pt ...  one transformer block per finished stage (frozen)
    norm-000.pt ...   output norm that goes with the stack up to that block
    stage-NNN/        the stage in training: latest.pt, best.pt, metrics.jsonl, evaluations.jsonl
    cache/            what the frozen stack outputs for every training window, stored on the drive

The cache is what keeps deep stacks affordable: from stage 1 on, a step reads the frozen stack's output
from disk and runs only the new block, so a stage costs the same however many blocks sit under it. The
frozen blocks are not even loaded while a stage trains. Advancing the cache by one block happens once,
at the start of the next stage.

    new     create an empty stack
    train   train the current stage (resumes it if it was started)
    freeze  turn the current stage into block-NNN.pt; the next `train` starts a new block
    grow    do both in a loop, unattended, until a block count or time budget is reached
    info    show what the stack holds

Same bytes, data slice, batch size, learning rate and validation batches as baseline.py.
"""
import argparse
import ctypes
import hashlib
import json
import math
import random
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import baseline
import earnest

CHUNK = 16  # windows pushed through a block at a time while the cache is advanced


class Stack(nn.Module):
    """Frozen embeddings + frozen blocks, then one trainable block and its output norm."""

    def __init__(self, cfg, frozen_blocks, stage, partial=False):
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.partial = partial  # True: frozen blocks are not loaded; only top() may be used
        self.token = nn.Embedding(256, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.frozen = nn.ModuleList(frozen_blocks)
        self.block = earnest.Block(cfg)
        self.norm = nn.LayerNorm(cfg.width)
        for module in (self.token, self.position):
            nn.init.normal_(module.weight, std=0.02)
        self.block.apply(earnest.Earnest.initialize)
        if stage > 0:
            # Identity start: the new block adds exactly zero at first, so a stage begins at the loss
            # the stack already reached instead of knocking it backwards.
            for parameter in (self.block.attn.out_proj.weight, self.block.attn.out_proj.bias,
                              self.block.ff[2].weight, self.block.ff[2].bias):
                nn.init.zeros_(parameter)

    def bottom(self, tokens):
        """Everything under the new block. Frozen from stage 1 on."""
        length = tokens.shape[1]
        if self.stage == 0:
            return self.token(tokens) + self.position(torch.arange(length))
        if self.partial:
            raise RuntimeError('Frozen blocks are not loaded in this model')
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        with torch.no_grad():
            x = self.token(tokens) + self.position(torch.arange(length))
            for block in self.frozen:
                x = block(x, mask)
        return x

    def top(self, x):
        """The new block, its norm and the output head, from the frozen stack's output."""
        length = x.shape[1]
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        return F.linear(self.norm(self.block(x, mask)), self.token.weight)

    def forward(self, tokens):
        return self.top(self.bottom(tokens))


# ----------------------------------------------------------------------------- files

def trainable_state(model):
    prefixes = ('block.', 'norm.', 'token.', 'position.') if model.stage == 0 else ('block.', 'norm.')
    return {key: value for key, value in model.state_dict().items() if key.startswith(prefixes)}


def save(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def read_meta(root):
    path = root / 'stack.json'
    if not path.exists():
        raise SystemExit(f'No stack at {root}. Create one with: stacked.py new --stack "{root}"')
    meta = json.loads(path.read_text(encoding='utf-8'))
    meta.setdefault('stride', meta['config']['context'])
    return meta


def write_meta(root, meta):
    temporary = root / 'stack.json.tmp'
    temporary.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    temporary.replace(root / 'stack.json')


def config_for(meta):
    return earnest.Config(width=meta['config']['width'], heads=meta['config']['heads'],
                          layers=1, context=meta['config']['context'])


def frozen_block(root, cfg, index):
    block = earnest.Block(cfg)
    block.load_state_dict(load(root / f'block-{index:03d}.pt'))
    for parameter in block.parameters():
        parameter.requires_grad_(False)
    return block


def build_model(root, meta, with_frozen=True):
    cfg = config_for(meta)
    stage = len(meta['blocks'])
    frozen = [frozen_block(root, cfg, index) for index in range(stage)] if with_frozen else []
    model = Stack(cfg, frozen, stage, partial=stage > 0 and not with_frozen)
    if stage > 0:
        embeddings = load(root / 'embed.pt')
        with torch.no_grad():
            model.token.weight.copy_(embeddings['token'])
            model.position.weight.copy_(embeddings['position'])
        model.norm.load_state_dict(load(root / f'norm-{stage - 1:03d}.pt'))
        for module in (model.token, model.position):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return model, stage


# ----------------------------------------------------------------------------- on-disk cache

def raw(tensor):
    tensor = tensor.contiguous()
    return ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size())


def read_rows(handle, rows, row_bytes, dtype, shape):
    out = []
    for row in rows:
        handle.seek(row * row_bytes)
        data = bytearray(handle.read(row_bytes))
        if len(data) != row_bytes:
            raise RuntimeError('Cache file is shorter than expected; delete the cache folder to rebuild it')
        out.append(torch.frombuffer(data, dtype=dtype).view(shape))
    return torch.stack(out)


def training_windows(documents, context, stride):
    return torch.stack([doc[start:start + context + 1] for doc in documents
                        for start in range(0, len(doc) - context, stride)]).to(torch.uint8)


def validation_windows(documents, context, batches=32):
    """Draw the same deterministic batches as baseline.validation_loss."""
    generator = random.Random(991)
    rows = []
    for _ in range(batches):
        x, y = earnest.batch(documents, context, 2, generator)
        rows.append(torch.cat([x, y[:, -1:]], dim=1))
    return torch.cat(rows).to(torch.uint8)


def ensure_cache(root, meta, stage, splits):
    """Bring cache/*.act up to 'output of the first `stage` frozen blocks' for every window."""
    cfg = config_for(meta)
    folder = root / 'cache'
    folder.mkdir(exist_ok=True)
    wanted = {'context': cfg.context, 'width': cfg.width, 'stride': meta['stride'],
              'source': meta['source_sha256'], 'windows': {name: len(rows) for name, rows in splits.items()}}
    path = folder / 'cache.json'
    info = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
    usable = (info is not None and 0 < info.get('blocks', -1) <= stage
              and all(info.get(key) == value for key, value in wanted.items()))
    have = info['blocks'] if usable else 0
    if have == stage:
        return wanted
    path.write_text(json.dumps(dict(wanted, blocks=-1)), encoding='utf-8')
    for name, rows in splits.items():
        (folder / f'{name}.tok').write_bytes(raw(rows))
    embeddings = load(root / 'embed.pt')
    blocks = None if 'crawl' in meta else [frozen_block(root, cfg, index) for index in range(have, stage)]
    mask = torch.ones(cfg.context, cfg.context, dtype=torch.bool).triu(1)
    row_bytes = cfg.context * cfg.width * 2
    total = sum(len(rows) for rows in splits.values())
    print(f'Cache: advancing {total:,} windows from {have} to {stage} frozen block(s), '
          f'{total * row_bytes / 1048576:,.0f} MB on disk', flush=True)
    started, done, announced = time.monotonic(), 0, 0
    with torch.no_grad():
        for name, rows in splits.items():
            old = (folder / f'{name}.act').open('rb') if have else None
            try:
                with (folder / f'{name}.act.tmp').open('wb') as out:
                    for first in range(0, len(rows), CHUNK):
                        count = min(CHUNK, len(rows) - first)
                        if old is None:
                            tokens = rows[first:first + count, :-1].long()
                            x = F.embedding(tokens, embeddings['token']) + embeddings['position']
                        else:
                            x = read_rows(old, range(first, first + count), row_bytes, torch.float16,
                                          (cfg.context, cfg.width)).float()
                        if blocks is None:
                            for index in range(have, stage):
                                x = frozen_block(root, cfg, index)(x, mask)
                        else:
                            for block in blocks:
                                x = block(x, mask)
                        out.write(raw(x.to(torch.float16)))
                        done += count
                        if done * 10 // total > announced:
                            announced = done * 10 // total
                            print(f'Cache: {announced * 10}% ({time.monotonic() - started:.0f} s)', flush=True)
            finally:
                if old is not None:
                    old.close()
    # Mark the cache unusable while the files are swapped, so an interruption can never leave a mixed one.
    path.write_text(json.dumps(dict(wanted, blocks=-1)), encoding='utf-8')
    for name in splits:
        (folder / f'{name}.act.tmp').replace(folder / f'{name}.act')
    path.write_text(json.dumps(dict(wanted, blocks=stage), indent=2), encoding='utf-8')
    return wanted


# ----------------------------------------------------------------------------- commands

def new(args):
    root = Path(args.stack).resolve()
    if (root / 'stack.json').exists():
        raise SystemExit(f'A stack already exists at {root}')
    root.mkdir(parents=True, exist_ok=True)
    stride = args.stride or args.context
    meta = {'config': {'width': args.width, 'heads': args.heads, 'context': args.context}, 'stride': stride,
            'revision': baseline.REVISION, 'train_bytes': args.train_bytes, 'source_sha256': None,
            'extra': [str(Path(item).resolve()) for item in args.extra], 'extra_repeat': args.extra_repeat,
            'created': datetime.now().isoformat(timespec='seconds'), 'blocks': []}
    if args.local_train:
        from crawl import read_corpus

        paths = {'train': str(Path(args.local_train).resolve()),
                 'validation': str(Path(args.local_validation).resolve())}
        _, _, fingerprints, _ = read_corpus(paths, args.context)
        meta['crawl'] = paths
        meta['source_sha256'] = fingerprints
        meta['validation_batches'] = 32
    write_meta(root, meta)
    per_block = sum(p.numel() for p in earnest.Block(config_for(meta)).parameters())
    cache_mb = args.train_bytes / stride * args.context * args.width * 2 / 1048576
    print(f'Created stack {root}: width {args.width}, heads {args.heads}, context {args.context}. '
          f'Each block adds {per_block:,} parameters. The cache will take about {cache_mb:,.0f} MB '
          f'(twice that briefly while it is advanced).', flush=True)


def fork(args):
    source = Path(args.source).resolve()
    target = Path(args.stack).resolve()
    meta = read_meta(source)
    if not meta['blocks']:
        raise SystemExit('Freeze at least one baseline block before forking')
    if target.exists() or source in target.parents:
        raise SystemExit('Choose a new stack directory outside the source stack')

    meta.update(
        source_sha256=None,
        train_bytes=args.train_bytes if args.train_bytes is not None else meta['train_bytes'],
        extra=[str(Path(item).resolve()) for item in args.extra],
        extra_repeat=args.extra_repeat,
        validation_bytes=args.validation_bytes,
        validation_batches=args.validation_batches,
        sample_seeds=args.sample_seed or [2026, 2027, 2028],
        created=datetime.now().isoformat(timespec='seconds'),
        parent={'stack': str(source), 'blocks': len(meta['blocks']),
                'source_sha256': meta['source_sha256']},
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix='.earnest-fork-') as temp:
        staged = Path(temp) / 'stack'
        staged.mkdir()
        shutil.copyfile(source / 'embed.pt', staged / 'embed.pt')
        for index, block in enumerate(meta['blocks']):
            for name in (block['file'], f'norm-{index:03d}.pt'):
                shutil.copyfile(source / name, staged / name)
        write_meta(staged, meta)
        if target.exists():
            raise SystemExit(f'A stack directory already exists at {target}')
        staged.rename(target)
    print(f'Forked {len(meta["blocks"])} frozen blocks into {target}; cache will build on first training run.')


_corpus = {}
EXTRA_SUFFIXES = {'.txt', '.md', '.json', '.jsonl'}


def local_documents(paths):
    """Your own text files as training documents. The last 10% of every file is held out and never trained on."""
    train, held, hashes, total = [], [], [], 0
    for item in paths:
        path = Path(item)
        files = sorted(f for f in path.rglob('*') if f.is_file() and f.suffix.lower() in EXTRA_SUFFIXES) if path.is_dir() else [path]
        for file in files:
            data = file.read_bytes()
            hashes.append(hashlib.sha256(data).hexdigest())
            total += len(data)
            cut = int(len(data) * 0.9)
            for part, bucket in ((data[:cut], train), (data[cut:], held)):
                for start in range(0, len(part), 4096):
                    piece = part[start:start + 4096]
                    if len(piece) >= 300:
                        bucket.append(torch.tensor(list(piece), dtype=torch.long))
    print(f'Read your own data: {len(hashes)} file(s), {total:,} bytes; {len(train)} training piece(s), '
          f'{len(held)} held out', flush=True)
    return train, held, hashes


def corpus(meta):
    """Training and validation documents as tensors; read once per process (grow runs many stages in one)."""
    if 'crawl' in meta:
        from crawl import read_corpus

        return read_corpus(meta['crawl'], meta['config']['context'])
    validation_bytes = meta.get('validation_bytes', 131072)
    key = (meta['train_bytes'], meta['revision'], validation_bytes,
           tuple(meta.get('extra', ())), meta.get('extra_repeat', 1))
    if key not in _corpus:
        train_text, train_record = baseline.read_split('train', meta['train_bytes'], meta['revision'])
        valid_text, valid_record = baseline.read_split('validation', validation_bytes, meta['revision'],
                                                       train_record['story_hashes'])
        training = baseline.tensorize(train_text)
        fingerprints = [train_record['received_sha256'], valid_record['received_sha256']]
        held = []
        if meta.get('extra'):
            own, held, hashes = local_documents(meta['extra'])
            if not own:
                raise SystemExit('The extra data has no usable text (need files of at least a few hundred bytes)')
            training += own * meta.get('extra_repeat', 1)
            fingerprints += hashes
        _corpus[key] = (training, baseline.tensorize(valid_text), fingerprints, held)
    return _corpus[key]


def rotation_focus(meta):
    rotation = meta.get('rotation')
    if rotation is None:
        return None
    index = (len(meta['blocks']) - rotation['start_depth']) // 3
    return rotation['phases'][index % len(rotation['phases'])]


def train(args):
    torch.set_num_threads(args.threads)
    root = Path(args.stack).resolve()
    meta = read_meta(root)
    focus = rotation_focus(meta)
    data = meta if focus is None else {**meta, 'extra': focus['extra'], 'extra_repeat': focus['repeat']}
    training, validation, fingerprints, held = corpus(data)
    if focus is None:
        provenance, key = meta, 'source_sha256'
    else:
        print(f'Rotation: block {len(meta["blocks"]) + 1}, focus {focus["name"]}.', flush=True)
        provenance, key = meta.setdefault('rotation_sources', {}), focus['name']
    if provenance.get(key) is None:
        provenance[key] = fingerprints
        write_meta(root, meta)
    elif provenance[key] != fingerprints:
        raise SystemExit('Source data differs from what this stack was started on; refusing to continue')
    data = {**data, 'source_sha256': fingerprints}

    stage = len(meta['blocks'])
    if 'crawl' in meta and stage > 0 and args.no_cache:
        raise SystemExit('Crawl uses the activation cache after the first block.')
    cached = stage > 0 and not args.no_cache
    validation_batches = meta.get('validation_batches', 32)
    crawl_cache = None
    if cached and 'crawl' in meta:
        context = meta['config']['context']
        splits = {'train': training_windows(training, context, meta['stride']),
                  'valid': validation_windows(validation, context, validation_batches)}
        crawl_cache = ensure_cache(root, data, stage, splits)['windows']
        del splits
    torch.manual_seed(42 + stage)
    model, stage = build_model(root, meta, with_frozen=not cached)
    cfg = model.cfg
    validation_batches = meta.get('validation_batches', 32)
    sample_seeds = meta.get('sample_seeds', (2026,))
    folder = root / f'stage-{stage:03d}'
    folder.mkdir(exist_ok=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    generator = random.Random(42 + stage)
    step = 0
    if (folder / 'latest.pt').exists():
        state = load(folder / 'latest.pt')
        model.load_state_dict(state['trainable'], strict=False)
        optimizer.load_state_dict(state['optimizer'])
        generator.setstate(state['python_rng'])
        torch.set_rng_state(state['torch_rng'])
        step = state['step']
        del state
    print(f'Stack: {root}', flush=True)
    print(f'Stage {stage}: training {sum(p.numel() for p in parameters):,} parameters over {stage} frozen block(s); '
          f'{"cached on disk" if cached else "live"}; resuming at step {step}', flush=True)

    files = {}
    if cached:
        if crawl_cache is None:
            splits = {'train': training_windows(training, cfg.context, meta['stride']),
                      'valid': validation_windows(validation, cfg.context, validation_batches)}
            counts = ensure_cache(root, data, stage, splits)['windows']
            del splits
        else:
            counts = crawl_cache
        files = {name: ((root / 'cache' / f'{name}.act').open('rb'), (root / 'cache' / f'{name}.tok').open('rb'))
                 for name in ('train', 'valid')}
    act_bytes, tok_bytes = cfg.context * cfg.width * 2, cfg.context + 1

    import feedback
    corrections = []
    feedback_every = meta.get("feedback_every", 20)
    approved = feedback.examples(root, cfg.context) if 'crawl' not in meta or meta.get('response_training') else []
    if approved:
        feature_model = model
        if cached:
            with torch.random.fork_rng():
                feature_model, _ = build_model(root, meta)
            feature_model.eval()
        for prompt, continuation in approved:
            tokens = torch.tensor([list(prompt + continuation)], dtype=torch.long)
            inputs, targets = tokens[:, :-1], tokens[:, len(prompt):]
            with torch.no_grad():
                features = feature_model.bottom(inputs) if cached else inputs
            corrections.append((features, targets, len(prompt) - 1))
        del feature_model
        print(f'Feedback: {len(corrections)} approved continuations; one in {feedback_every} updates.', flush=True)

    def cached_loss(name, rows):
        acts, tokens = files[name]
        x = read_rows(acts, rows, act_bytes, torch.float16, (cfg.context, cfg.width)).float()
        y = read_rows(tokens, rows, tok_bytes, torch.uint8, (tok_bytes,))[:, 1:].long()
        return F.cross_entropy(model.top(x).reshape(-1, 256), y.reshape(-1))

    def validate():
        if not cached:
            return baseline.validation_loss(model, validation, validation_batches)
        model.eval()
        with torch.no_grad():
            values = [cached_loss('valid', (2 * index, 2 * index + 1)).item()
                      for index in range(validation_batches)]
        model.train()
        return sum(values) / len(values) / math.log(2)

    def checkpoint(name):
        save(folder / name, {'stage': stage, 'config': meta['config'], 'trainable': trainable_state(model),
                             'optimizer': optimizer.state_dict(), 'step': step,
                             'torch_rng': torch.get_rng_state(), 'python_rng': generator.getstate()})

    best = float('inf')
    evaluations = folder / 'evaluations.jsonl'
    if evaluations.exists():
        prior = [json.loads(line) for line in evaluations.read_text(encoding='utf-8').splitlines()]
        if prior:
            best = min(row['validation_bpb'] for row in prior)

    def assess(extra=None):
        nonlocal best
        value = validate()
        baseline.append_json(folder / 'evaluations.jsonl', dict({'step': step, 'validation_bpb': value}, **(extra or {})))
        checkpoint('latest.pt')
        if value < best:
            best = value
            checkpoint('best.pt')
        print(f'Stage {stage} step {step}: validation {value:.4f} bits/byte', flush=True)
        return value

    def live_samples():
        # Stage 0 can write text straight from the model in training, so every evaluation carries samples.
        # Later stages train from the cache without the frozen blocks loaded; they get samples once per round.
        return None if cached else {'samples': baseline.samples(model, sample_seeds)}

    initial = assess(None if 'crawl' in meta else live_samples())
    start = time.monotonic()
    first_step = last_assessed = step
    status = 'completed'
    stop_file = getattr(args, 'stop_file', None)
    try:
        for _ in range(args.steps):
            if stop_file is not None and stop_file.exists():
                status = 'interrupted'
                break
            if time.monotonic() - start >= args.max_minutes * 60:
                status = 'time budget reached'
                break
            optimizer.zero_grad(set_to_none=True)
            if corrections and step % feedback_every == 0:
                features, targets, start_at = corrections[(step // feedback_every) % len(corrections)]
                logits = model.top(features) if cached else model(features)
                loss = F.cross_entropy(logits[:, start_at:].reshape(-1, 256), targets.reshape(-1))
            elif cached:
                loss = cached_loss('train', [generator.randrange(counts['train']) for _ in range(args.batch_size)])
            else:
                loss = earnest.loss_for(model, *earnest.batch(training, cfg.context, args.batch_size, generator))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            step += 1
            if step % 50 == 0:
                row = {'step': step, 'train_bpb': loss.item() / math.log(2),
                       'elapsed_seconds': time.monotonic() - start}
                baseline.append_json(folder / 'metrics.jsonl', row)
                print(json.dumps(row), flush=True)
                checkpoint('latest.pt')
            if step % 200 == 0:
                assess(None if 'crawl' in meta else live_samples())
                last_assessed = step
    except KeyboardInterrupt:
        status = 'interrupted'
    seconds = time.monotonic() - start

    # End of the invocation: load the whole stack once for an exact (uncached) validation and text samples.
    if 'crawl' in meta:
        exact, own = None, None
        final = assess()
    else:
        full = model
        if cached:
            with torch.random.fork_rng():
                full, _ = build_model(root, meta, with_frozen=True)
            full.load_state_dict(trainable_state(model), strict=False)
        exact = baseline.validation_loss(full, validation, validation_batches)
        own = baseline.validation_loss(full, held, validation_batches) if held else None
        if own is not None:
            print(f'Stage {stage} step {step}: held-out part of your own data {own:.4f} bits/byte', flush=True)
        final = assess({'exact_validation_bpb': exact, 'own_data_validation_bpb': own,
                        'samples': baseline.samples(full, sample_seeds)})
    result = {'status': status, 'stage': stage, 'step': step, 'updates_this_invocation': step - first_step,
              'cached': cached, 'initial_validation_bpb': initial, 'final_validation_bpb': final,
              'exact_validation_bpb': exact, 'own_data_validation_bpb': own, 'best_validation_bpb': best,
              'seconds': seconds,
              'seconds_per_step': seconds / max(1, step - first_step), 'folder': str(folder)}
    for acts, tokens in files.values():
        acts.close()
        tokens.close()
    (folder / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    if meta.get('grammar_foundation') and step:
        import grammar_foundation
        scores = {'frozen': grammar_foundation.evaluate(root),
                  'candidate': grammar_foundation.evaluate(root, folder / 'latest.pt')}
        snapshot = folder / f'grammar-{step:06d}.json'
        snapshot.write_text(json.dumps(scores, indent=2), encoding='utf-8')
        print(f'Grammar holdout: {scores["frozen"]["correct"]}/{scores["frozen"]["total"]} frozen, '
              f'{scores["candidate"]["correct"]}/{scores["candidate"]["total"]} candidate.', flush=True)

    if 'crawl' in meta and status == 'completed' and step > first_step:
        # Release the training model before the review loads its checkpoint.
        del optimizer, parameters, model
        import crawl

        crawl.review(root, 'latest', verbose=False)

    return result


def grow(args):
    """Unattended building: train the current block until it stops paying, freeze it, start the next."""
    root = Path(args.stack).resolve()
    continuous = getattr(args, 'continuous', False)
    stop_file = getattr(args, 'stop_file', None)
    meta = read_meta(root)
    if continuous and 'crawl' in meta:
        try:
            if meta.get('crawl_selection'):
                import crawl_selection

                def trial(**parameters):
                    return train(argparse.Namespace(**parameters,no_cache=False))

                crawl_selection.run(root,args,stop_file,trial)
                return
            while stop_file is None or not stop_file.exists():
                result = train(argparse.Namespace(stack=root, steps=args.round_steps, max_minutes=args.round_minutes,
                                                  batch_size=args.batch_size, threads=args.threads,
                                                  lr=args.lr, no_cache=False, stop_file=stop_file))
                if result['status'] == 'interrupted':
                    break
        finally:
            if stop_file is not None:
                stop_file.unlink(missing_ok=True)
        print('Crawl stopped. The last checkpoint is saved.', flush=True)
        return
    deadline = math.inf if continuous else time.monotonic() + args.hours * 3600
    while True:
        meta = read_meta(root)
        stage = len(meta['blocks'])
        focus = rotation_focus(meta)
        if not getattr(args, 'continuous', False) and stage >= args.blocks:
            print(f'Grow: the stack has {stage} block(s), the target. Done.', flush=True)
            return
        previous = meta['blocks'][-1]['validation_bpb'] if meta['blocks'] else None
        best = None
        while True:
            minutes = (deadline - time.monotonic()) / 60
            if minutes < 0.5:
                print('Grow: time budget reached. The stage in training is saved; run grow again to continue.', flush=True)
                return
            result = train(argparse.Namespace(stack=args.stack, steps=args.round_steps,
                                              max_minutes=min(args.round_minutes, minutes), batch_size=args.batch_size,
                                              threads=args.threads, lr=args.lr, no_cache=getattr(args, 'no_cache', False)))
            if result['status'] == 'interrupted':
                print('Grow: interrupted. The stage in training is saved; run grow again to continue.', flush=True)
                return
            gain = best - result['best_validation_bpb'] if best is not None else None
            best = result['best_validation_bpb']
            if gain is not None:
                print(f'Grow: stage {stage} gained {gain:.4f} bits/byte this round', flush=True)
            round_gain = args.min_gain if args.round_gain is None else args.round_gain
            # General validation can plateau before the continuity objective does.
            plateau = focus is None and 'continuity_gate' not in meta and gain is not None and gain < round_gain
            if result['step'] >= args.stage_steps or plateau:
                break
        if stage and args.accept_gain is not None:
            evaluations = root / f'stage-{stage:03d}' / 'evaluations.jsonl'
            with evaluations.open(encoding='utf-8') as handle:
                identity = json.loads(next(handle))
            if identity['step'] != 0:
                raise SystemExit('Strict acceptance requires the stage-zero evaluation for this block')
            improvement = identity['validation_bpb'] - best
            if improvement < args.accept_gain:
                print(f'Grow: block {stage + 1} gained {improvement:.4f} bits/byte; '
                      f'required {args.accept_gain:.4f}. Checkpoints saved without freezing the block.', flush=True)
                return
        if not freeze(argparse.Namespace(stack=args.stack, use='best')):
            if getattr(args, 'continuous', False) and focus is not None:
                print('Continuing the same block from its last checkpoint.', flush=True)
                continue
            return
        if (focus is None and 'continuity_gate' not in meta and args.accept_gain is None
                and previous is not None and previous - best < args.min_gain):
            print(f'Grow: block {stage} improved the whole stack by only {previous - best:.4f} bits/byte, so it has '
                  f'levelled off. Stopping: more blocks will not help at this width and amount of data.', flush=True)
            return


def freeze(args):
    root = Path(args.stack).resolve()
    meta = read_meta(root)
    stage = len(meta['blocks'])
    folder = root / f'stage-{stage:03d}'
    source = folder / 'best.pt' if args.use == 'best' and (folder / 'best.pt').exists() else folder / 'latest.pt'
    if not source.exists():
        raise SystemExit(f'Stage {stage} has no checkpoint yet; run `train` first')
    if 'crawl' in meta:
        import crawl

        source = crawl.approved_checkpoint(folder, args.use, getattr(args, 'coherent', None), stage)
        if source is None:
            return False
    state = load(source)
    weights = state['trainable']
    rows = [json.loads(line) for line in (folder / 'evaluations.jsonl').read_text(encoding='utf-8').splitlines()]
    at_step = [row['validation_bpb'] for row in rows if row['step'] == state['step']]
    if 'crawl' in meta and not crawl.validation_ok(rows, state['step']):
        print('Phase 0: held-out BPB exceeded the step-zero score by more than 0.01, or no updates were recorded.')
        return False
    gate = None if 'crawl' in meta else meta.get('continuity_gate')
    focus = rotation_focus(meta)
    if gate is not None or focus is not None:
        import continuity

        if stage == 0 or not rows or rows[0]['step'] != 0 or not at_step:
            raise SystemExit('Continuity gate requires a frozen parent and validation for both checkpoints')
        model, _ = build_model(root, meta)
        parent = continuity.score(model, frozen=True)
        candidates = [(source, state)]
        latest = folder / 'latest.pt'
        if source != latest and latest.exists():
            latest_state = load(latest)
            if latest_state['step'] != state['step']:
                candidates.append((latest, latest_state))
        reports = []
        for path, checkpoint in candidates:
            evaluation = [row['validation_bpb'] for row in rows if row['step'] == checkpoint['step']]
            if not evaluation:
                continue
            model.load_state_dict(checkpoint['trainable'], strict=False)
            candidate = continuity.score(model, frozen=False)
            general_gain = rows[0]['validation_bpb'] - evaluation[-1]
            if gate is not None:
                accepted = continuity.passes_gate(parent, candidate, general_gain,
                                                  gate['min_margin_bpb'], gate['max_general_regression_bpb'])
            else:
                accepted = (checkpoint['step'] > 0 and general_gain >= -0.01
                            and candidate['passes'] >= parent['passes']
                            and candidate['margin_bpb'] >= parent['margin_bpb'] - 0.005
                            and (general_gain >= 0.001 or candidate['passes'] > parent['passes']
                                 or candidate['margin_bpb'] - parent['margin_bpb'] >= 0.001))
            reports.append({'candidate': candidate, 'general_gain_bpb': general_gain,
                            'checkpoint': path.name, 'step': checkpoint['step'], 'accepted': accepted})
        selected = max(reports, key=lambda report: (report['accepted'], report['candidate']['passes'],
                                                     report['candidate']['margin_bpb']))
        candidate = selected['candidate']
        general_gain = selected['general_gain_bpb']
        accepted = selected['accepted']
        report = {'parent': parent, **selected, 'evaluated_checkpoints': reports}
        if focus is not None:
            report['focus'] = focus['name']
        (folder / 'continuity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'Continuity: {parent["passes"]}/{parent["pairs"]} → {candidate["passes"]}/{candidate["pairs"]}; '
              f'margin gain {candidate["margin_bpb"] - parent["margin_bpb"]:+.4f} BPB; '
              f'general gain {general_gain:+.4f} BPB.', flush=True)
        if not accepted:
            print('Continuity gate: block held for review; checkpoints preserved.', flush=True)
            return False
        source, state = next((path, checkpoint) for path, checkpoint in candidates
                             if path.name == selected['checkpoint'])
        weights = state['trainable']
        at_step = [row['validation_bpb'] for row in rows if row['step'] == state['step']]
    save(root / f'block-{stage:03d}.pt', {k[len('block.'):]: v for k, v in weights.items() if k.startswith('block.')})
    save(root / f'norm-{stage:03d}.pt', {k[len('norm.'):]: v for k, v in weights.items() if k.startswith('norm.')})
    if stage == 0:
        save(root / 'embed.pt', {'token': weights['token.weight'], 'position': weights['position.weight']})
    meta['blocks'].append({'file': f'block-{stage:03d}.pt', 'steps': state['step'], 'from': source.name,
                           'validation_bpb': at_step[-1] if at_step else None,
                           'parameters': sum(v.numel() for k, v in weights.items() if k.startswith('block.')),
                           'frozen': datetime.now().isoformat(timespec='seconds')})
    write_meta(root, meta)
    print(f'Froze stage {stage} from {source.name} (step {state["step"]}) into block-{stage:03d}.pt. '
          f'The stack now has {stage + 1} block(s); the next `train` starts block {stage + 1}.', flush=True)
    return True


def info(args):
    root = Path(args.stack).resolve()
    meta = read_meta(root)
    cfg = meta['config']
    print(f'Stack {root}\nwidth {cfg["width"]}, heads {cfg["heads"]}, context {cfg["context"]}, stride {meta["stride"]}; '
          f'{len(meta["blocks"])} frozen block(s), {sum(b["parameters"] for b in meta["blocks"]):,} parameters in blocks')
    for index, block in enumerate(meta['blocks']):
        size = (root / block['file']).stat().st_size / 1024
        value = f'{block["validation_bpb"]:.4f}' if block['validation_bpb'] is not None else 'n/a'
        print(f'  block {index}: {block["parameters"]:,} parameters, {size:,.0f} KB on disk, '
              f'{block["steps"]} steps, validation {value} bits/byte')
    folder = root / f'stage-{len(meta["blocks"]):03d}'
    if (folder / 'evaluations.jsonl').exists():
        last = json.loads((folder / 'evaluations.jsonl').read_text(encoding='utf-8').splitlines()[-1])
        print(f'  stage {len(meta["blocks"])} in training: step {last["step"]}, '
              f'validation {last["validation_bpb"]:.4f} bits/byte (not frozen yet)')
    else:
        print(f'  stage {len(meta["blocks"])}: not started')
    cache = root / 'cache' / 'cache.json'
    if cache.exists():
        state = json.loads(cache.read_text(encoding='utf-8'))
        size = sum(f.stat().st_size for f in cache.parent.iterdir() if f.is_file()) / 1048576
        print(f'  cache: {state["windows"]["train"]:,} training windows through {state["blocks"]} block(s), {size:,.0f} MB')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    creating = sub.add_parser('new')
    creating.add_argument('--stack', required=True)
    creating.add_argument('--width', type=int, default=64)
    creating.add_argument('--heads', type=int, default=4)
    creating.add_argument('--context', type=int, default=256)
    creating.add_argument('--stride', type=int, default=0, help='bytes between cached training windows (default: context)')
    creating.add_argument('--train-bytes', type=int, default=1048576)
    creating.add_argument('--extra', action='append', default=[],
                          help='your own text file or folder (.txt .md .json .jsonl) to train on beside TinyStories; repeatable')
    creating.add_argument('--extra-repeat', type=int, default=1, help='how many times your own data counts in the mix')
    creating.add_argument('--local-train')
    creating.add_argument('--local-validation')
    forking = sub.add_parser('fork', help='start a new data phase from an existing frozen stack')
    forking.add_argument('--source', required=True)
    forking.add_argument('--stack', required=True)
    forking.add_argument('--train-bytes', type=int)
    forking.add_argument('--extra', action='append', default=[])
    forking.add_argument('--extra-repeat', type=int, default=1)
    forking.add_argument('--validation-bytes', type=int, default=524288)
    forking.add_argument('--validation-batches', type=int, default=128)
    forking.add_argument('--sample-seed', type=int, action='append')
    training = sub.add_parser('train')
    training.add_argument('--stack', required=True)
    training.add_argument('--steps', type=int, default=1000)
    training.add_argument('--max-minutes', type=float, default=10)
    training.add_argument('--batch-size', type=int, default=2)
    training.add_argument('--threads', type=int, default=1)
    training.add_argument('--lr', type=float, default=0.001)
    training.add_argument('--no-cache', action='store_true', help='run the frozen blocks live instead of reading the cache')
    growing = sub.add_parser('grow')
    growing.add_argument('--stack', required=True)
    growing.add_argument('--blocks', type=int, required=True, help='stop when the stack has this many frozen blocks')
    growing.add_argument('--hours', type=float, default=8, help='overall time budget')
    growing.add_argument('--continuous', action='store_true', help='keep training until stopped')
    growing.add_argument('--round-steps', type=int, default=1000, help='updates between progress checks')
    growing.add_argument('--stop-file', type=Path, help=argparse.SUPPRESS)
    growing.add_argument('--round-minutes', type=float, default=30)
    growing.add_argument('--stage-steps', type=int, default=6000, help='most updates one block may get')
    growing.add_argument('--min-gain', type=float, default=0.01,
                         help='freeze a block when a round improves validation by less than this (bits/byte)')
    growing.add_argument('--round-gain', type=float, help='within-block plateau threshold in bits/byte')
    growing.add_argument('--accept-gain', type=float,
                         help='require this gain over the identity start before freezing a block')
    growing.add_argument('--batch-size', type=int, default=2)
    growing.add_argument('--threads', type=int, default=1)
    growing.add_argument('--lr', type=float, default=0.001)
    growing.add_argument('--no-cache', action='store_true', help='run the frozen blocks live instead of building the activation cache')
    freezing = sub.add_parser('freeze')
    freezing.add_argument('--stack', required=True)
    freezing.add_argument('--use', choices=('best', 'latest'), default='best')
    freezing.add_argument('--coherent', type=int, help='number of readable raw samples in the saved crawl review')
    reviewing = sub.add_parser('crawl-review')
    reviewing.add_argument('--stack', required=True)
    reviewing.add_argument('--use', choices=('best', 'latest'), default='best')
    showing = sub.add_parser('info')
    showing.add_argument('--stack', required=True)
    args = parser.parse_args()
    if args.command == 'new':
        if bool(args.local_train) != bool(args.local_validation):
            parser.error('Supply both --local-train and --local-validation')
        if args.local_train and (args.extra or args.context != 256):
            parser.error('Crawl requires 256-byte context and explicit local sources without --extra')
        if args.width < 8 or args.heads < 1 or args.width % args.heads:
            parser.error('--width must be at least 8 and divisible by --heads')
        if not 16 <= args.context <= 256:
            parser.error('--context must be 16 to 256 (stories shorter than 257 bytes are skipped)')
        if args.stride < 0 or args.train_bytes < 65536 or args.extra_repeat < 1:
            parser.error('--stride must not be negative, --train-bytes at least 65536, --extra-repeat at least 1')
        for item in args.extra:
            if not Path(item).exists():
                parser.error(f'--extra path does not exist: {item}')
        new(args)
    elif args.command == 'fork':
        if ((args.train_bytes is not None and args.train_bytes < 65536)
                or args.validation_bytes < 65536 or args.validation_batches < 1 or args.extra_repeat < 1
                or any(not 0 <= seed < 2**64 for seed in args.sample_seed or ())):
            parser.error('Invalid fork data budget, validation batch count, repeat count or sample seed')
        for item in args.extra:
            if not Path(item).exists():
                parser.error(f'--extra path does not exist: {item}')
        fork(args)
    elif args.command == 'train':
        if args.steps < 1 or args.max_minutes <= 0 or args.batch_size < 1 or args.threads < 1 or args.lr <= 0:
            parser.error('Budgets, batch size, threads and learning rate must be positive')
        train(args)
    elif args.command == 'grow':
        if (args.blocks < 1 or args.hours <= 0 or args.round_steps < 1 or args.round_minutes <= 0 or args.stage_steps < 1
                or args.batch_size < 1 or args.threads < 1 or args.lr <= 0
                or any(not math.isfinite(gain) or gain < 0
                       for gain in (args.min_gain, args.round_gain, args.accept_gain) if gain is not None)):
            parser.error('grow budgets must be positive')
        grow(args)
    elif args.command == 'freeze':
        freeze(args)
    elif args.command == 'crawl-review':
        import crawl

        crawl.review(Path(args.stack).resolve(), args.use)
    else:
        info(args)


if __name__ == '__main__':
    main()
