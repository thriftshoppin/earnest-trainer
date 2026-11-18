"""Local-only corpus and explicit human promotion review for Phase 0."""
import hashlib
import json
import math
from pathlib import Path

PROMPTS = ('The sun is', 'A pencil makes', 'The dog can', 'I put the', 'When it rains,')


def read_corpus(paths, context):
    import torch

    documents, fingerprints = [], []
    for split in ('train', 'validation'):
        path = Path(paths[split])
        raw = path.read_bytes()
        data = raw.decode('utf-8-sig').encode('utf-8')
        if len(data) <= context:
            raise ValueError(f'{split} must contain more than {context} UTF-8 bytes: {path}')
        documents.append([torch.tensor(list(data), dtype=torch.long)])
        fingerprints.append(hashlib.sha256(raw).hexdigest())
    if fingerprints[0] == fingerprints[1]:
        raise ValueError('Training and validation must be distinct files with distinct contents.')
    return documents[0], documents[1], fingerprints, []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root, use, verbose=True):
    import stacked
    from prompt_probe import complete

    meta = stacked.read_meta(root)
    if 'crawl' not in meta:
        raise SystemExit('This stack is not a crawl stack.')
    stage = len(meta['blocks'])
    folder = root / f'stage-{stage:03d}'
    path = folder / f'{use}.pt'
    if not path.is_file():
        raise SystemExit(f'Checkpoint does not exist: {path}')
    snapshot = folder / f'crawl-review-{use}.pt'
    import shutil
    temporary = snapshot.with_suffix('.tmp')
    shutil.copyfile(path, temporary)
    temporary.replace(snapshot)
    prompts = meta['crawl'].get('prompts', PROMPTS)
    state = stacked.load(snapshot)
    seeds = (2026, 2027 + state['step']) if meta.get('crawl_selection') else (2026,)
    samples = [complete(root, prompt, seed, 64, use_dictionary=False, checkpoint=snapshot)
               | {'prompt': prompt} for seed in seeds for prompt in prompts]
    report = {'checkpoint': snapshot.name, 'sha256': digest(snapshot),
              'step': state['step'], 'stage': stage, 'samples': samples}
    output = folder / f'crawl-review-{use}.json'
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(output)
    if verbose:
        print(json.dumps(report, indent=2))
        print(f'Review the raw samples before using freeze --use {use} --coherent COUNT; '
              'count only readable, fitting multi-word continuations.')
    else:
        print(f'Raw crawl review: {len(samples)} samples at update {state["step"]:,}. See the sample panel.', flush=True)
    return report


def approved_checkpoint(folder, use, count, stage):
    path = folder / f'crawl-review-{use}.json'
    if count is None or not path.is_file():
        print(f'Phase 0: use crawl-review --use {use}, then review the raw output. Candidate stays trainable.')
        return None
    report = json.loads(path.read_text(encoding='utf-8'))
    snapshot = folder / f'crawl-review-{use}.pt'
    if report['stage'] != stage or not snapshot.is_file() or report['sha256'] != digest(snapshot):
        raise SystemExit('The reviewed checkpoint changed. Create a fresh crawl review.')
    total = len(report['samples'])
    if not total // 2 < count <= total:
        print(f'Phase 0: coherence {count}/{total}; a majority is required. Candidate stays trainable.')
        return None
    return snapshot


def validation_ok(rows, step):
    values = [row['validation_bpb'] for row in rows if row['step'] == step]
    return (step > 0 and rows and rows[0]['step'] == 0 and values
            and math.isfinite(values[-1]) and values[-1] <= rows[0]['validation_bpb'] + 0.01)
