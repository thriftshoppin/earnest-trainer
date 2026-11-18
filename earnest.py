"""Small CPU-only, from-scratch language-model trainer for Earnest."""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    width: int = 64
    heads: int = 4
    layers: int = 2
    context: int = 64


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        self.attn = nn.MultiheadAttention(cfg.width, cfg.heads, dropout=0, batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                nn.Linear(cfg.width * 4, cfg.width))

    def forward(self, x, mask):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class Earnest(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token = nn.Embedding(256, cfg.width)
        self.position = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, 256, bias=False)
        self.head.weight = self.token.weight
        self.apply(self.initialize)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens):
        length = tokens.shape[1]
        x = self.token(tokens) + self.position(torch.arange(length))
        mask = torch.ones(length, length, dtype=torch.bool).triu(1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.norm(x))


class TextOnly(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style'):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ('script', 'style'):
            self.skip = max(0, self.skip - 1)

    def handle_data(self, text):
        if not self.skip and text.strip():
            self.parts.append(text.strip())


def fetch(url, limit):
    if urlparse(url).scheme != 'https':
        raise ValueError('Sources must use HTTPS')
    req = Request(url, headers={'User-Agent': 'EarnestTrainer/0.1', 'Accept-Encoding': 'identity'})
    with urlopen(req, timeout=30) as response:
        if urlparse(response.url).scheme != 'https':
            raise ValueError('Refusing non-HTTPS redirect')
        data = response.read(limit + 1)
        if len(data) > limit:
            raise ValueError(f'Source exceeds {limit} bytes: {url}')
        return data, response.headers.get('Content-Type', ''), response.url


def source_urls(url, max_files):
    parsed = urlparse(url)
    bits = parsed.path.strip('/').split('/')
    if parsed.hostname == 'github.com' and len(bits) == 2:
        owner, repo = bits
        repo = repo.removesuffix('.git')
        api = f'https://api.github.com/repos/{quote(owner)}/{quote(repo)}'
        metadata = json.loads(fetch(api + '/commits/HEAD', 2_000_000)[0])
        revision = metadata['sha']
        tree = json.loads(fetch(api + f'/git/trees/{revision}?recursive=1', 4_000_000)[0])
        if tree.get('truncated'):
            raise ValueError('Repository tree truncated; supply individual raw file URLs')
        extensions = {'.py', '.md', '.txt', '.rst', '.js', '.ts', '.rs', '.c', '.cpp', '.h'}
        files = [item['path'] for item in tree['tree'] if item['type'] == 'blob'
                 and Path(item['path']).suffix.lower() in extensions and item.get('size', 0) <= 262144]
        files.sort(key=lambda p: (not Path(p).name.lower().startswith('readme'), p))
        if not files:
            raise ValueError('No supported text files found')
        return [f'https://raw.githubusercontent.com/{owner}/{repo}/{revision}/{quote(p, safe="/")}'
                for p in files[:max_files]]
    return [url]


def read_sources(urls, max_files=8, max_bytes=1048576):
    documents, records = [], []
    remaining = max_bytes
    for source in urls:
        for url in source_urls(source, max_files):
            if remaining < 256:
                break
            data, content_type, resolved = fetch(url, min(remaining, 262144))
            remaining -= len(data)
            if b'\x00' in data:
                raise ValueError(f'Binary source is unsupported: {url}')
            text = data.decode('utf-8', errors='strict')
            if 'html' in content_type:
                parser = TextOnly()
                parser.feed(text)
                text = '\n'.join(parser.parts)
            elif not ('text/' in content_type or 'json' in content_type or 'octet-stream' in content_type):
                raise ValueError(f'Unsupported source type {content_type}: {url}')
            documents.append(text.encode('utf-8'))
            records.append({'url': resolved, 'sha256': hashlib.sha256(data).hexdigest(),
                            'bytes_received': len(data), 'training_text_bytes': len(documents[-1])})
    return documents, records


def split_documents(documents, context):
    train, valid = [], []
    for data in documents:
        boundary = int(len(data) * 0.85)
        if boundary > context and len(data) - boundary > context:
            train.append(torch.tensor(list(data[:boundary]), dtype=torch.long))
            valid.append(torch.tensor(list(data[boundary:]), dtype=torch.long))
    if not train:
        raise ValueError('Need a longer text source (at least about 500 bytes for default context)')
    return train, valid


def batch(documents, context, size, generator):
    weights = [len(doc) - context for doc in documents]
    chunks = []
    for _ in range(size):
        doc = generator.choices(documents, weights=weights, k=1)[0]
        start = generator.randrange(len(doc) - context)
        chunks.append(doc[start:start + context + 1])
    value = torch.stack(chunks)
    return value[:, :-1], value[:, 1:]


def loss_for(model, x, y):
    return F.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))


@torch.no_grad()
def evaluate(model, valid, batch_size):
    model.eval()
    generator = random.Random(991)
    values = [loss_for(model, *batch(valid, model.cfg.context, batch_size, generator)).item()
              for _ in range(4)]
    model.train()
    return sum(values) / len(values) / math.log(2)


def save_checkpoint(path, model, optimizer, step, generator, records):
    payload = {'config': asdict(model.cfg), 'model': model.state_dict(),
               'optimizer': optimizer.state_dict(), 'step': step,
               'torch_rng': torch.get_rng_state(), 'python_rng': generator.getstate(),
               'sources': records}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    generator = random.Random(42)
    cfg = Config()
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if saved:
        cfg = Config(**saved['config'])
    if args.smoke:
        documents = [('Earnest learns one byte at a time. Small computers can run small experiments.\n'
                      * 100).encode()]
        records = [{'source': 'built-in synthetic smoke fixture',
                    'sha256': hashlib.sha256(documents[0]).hexdigest()}]
    else:
        urls = list(args.url)
        if args.sources:
            urls.extend(json.loads(Path(args.sources).read_text(encoding='utf-8-sig')))
        if not urls:
            raise ValueError('Provide --url HTTPS_URL or --sources sources.json')
        documents, records = read_sources(urls)
    training, validation = split_documents(documents, cfg.context)
    del documents
    model = Earnest(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    step = 0
    if saved:
        if saved['sources'] != records and not args.allow_new_sources:
            raise ValueError('Sources changed; use --allow-new-sources for deliberate continued learning')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
        torch.set_rng_state(saved['torch_rng'])
        generator.setstate(saved['python_rng'])
        del saved
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'latest.pt'
    if checkpoint.exists() and not args.resume:
        raise ValueError('Output checkpoint already exists: use --resume or a new --out directory')
    metadata = {'config': asdict(cfg), 'parameters': sum(p.numel() for p in model.parameters()),
                'torch': torch.__version__, 'threads': args.threads, 'sources': records,
                'train_bytes': sum(len(d) for d in training),
                'validation_bytes': sum(len(d) for d in validation), 'smoke': args.smoke}
    (output / 'run.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata), flush=True)
    initial = evaluate(model, validation, args.batch_size)
    print(f'Initial validation: {initial:.4f} bits/byte', flush=True)
    start = time.monotonic()
    first_step = step
    interrupted = False
    try:
        for _ in range(args.steps):
            x, y = batch(training, cfg.context, args.batch_size, generator)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(model, x, y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 10 == 0 or step == first_step + args.steps:
                row = {'step': step, 'train_bits_per_byte': loss.item() / math.log(2),
                       'elapsed_seconds': time.monotonic() - start}
                with (output / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
            if step % 50 == 0:
                save_checkpoint(checkpoint, model, optimizer, step, generator, records)
    except KeyboardInterrupt:
        interrupted = True
        print('Interrupted; saving current state.', flush=True)
    save_checkpoint(checkpoint, model, optimizer, step, generator, records)
    final = evaluate(model, validation, args.batch_size)
    result = {'step': step, 'initial_validation_bpb': initial, 'final_validation_bpb': final,
              'seconds': time.monotonic() - start, 'checkpoint': str(checkpoint.resolve()),
              'interrupted': interrupted}
    (output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


@torch.no_grad()
def generate(args):
    torch.set_num_threads(args.threads)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Earnest(Config(**saved['config']))
    model.load_state_dict(saved['model'])
    model.eval()
    result = bytearray(args.prompt.encode('utf-8') or b' ')
    for _ in range(args.bytes):
        x = torch.tensor([list(result[-model.cfg.context:])], dtype=torch.long)
        probs = (model(x)[0, -1] / args.temperature).softmax(-1)
        result.append(torch.multinomial(probs, 1).item())
    print(result.decode('utf-8', errors='replace'))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train')
    training.add_argument('--url', action='append', default=[])
    training.add_argument('--sources')
    training.add_argument('--smoke', action='store_true')
    training.add_argument('--steps', type=int, default=100)
    training.add_argument('--batch-size', type=int, default=2)
    training.add_argument('--threads', type=int, default=1)
    training.add_argument('--out', default='runs/earnest')
    training.add_argument('--resume')
    training.add_argument('--allow-new-sources', action='store_true')
    sampling = sub.add_parser('generate')
    sampling.add_argument('--checkpoint', required=True)
    sampling.add_argument('--prompt', default='Earnest')
    sampling.add_argument('--bytes', type=int, default=200)
    sampling.add_argument('--temperature', type=float, default=0.8)
    sampling.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.command == 'train':
        if args.steps < 1 or args.batch_size < 1:
            parser.error('--steps and --batch-size must be positive')
        train(args)
    else:
        if args.bytes < 1 or args.temperature <= 0:
            parser.error('--bytes and --temperature must be positive')
        generate(args)


if __name__ == '__main__':
    main()
