"""Generate a continuation from the latest fully frozen Earnest stack."""
import argparse
from bisect import bisect_left
import json
import re
import string
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

import stacked


class EnglishLexicon:
    source = Path(__file__).resolve().parent / 'lexicon' / 'cmudict.dict'
    word = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
    prefix = re.compile(rb"[a-z]+(?:'[a-z]+)*'?\Z")
    alphabet = string.ascii_lowercase + "'"
    boundaries = tuple(map(ord, string.digits + string.punctuation.replace("'", "") + " ")) + (9, 10)
    starters = boundaries + tuple(map(ord, string.ascii_letters + "'"))

    def __init__(self, prompt):
        words = {word.lower() for word in self.word.findall(prompt)}
        with self.source.open(encoding='utf-8') as entries:
            for line in entries:
                entry = line.partition(' ')[0].partition('(')[0]
                if self.word.fullmatch(entry):
                    words.add(entry.lower())
        if len(words) < 50000:
            raise ValueError('The English dictionary is incomplete.')
        self.words = words
        self.sorted_words = sorted(words)

    def current(self, tokens):
        match = self.prefix.search(bytes(tokens).lower())
        return match.group().decode('ascii') if match else ''

    def choices(self, tokens):
        current = self.current(tokens)
        if not current:
            return self.starters
        choices = list(self.boundaries) if current in self.words else []
        for letter in self.alphabet:
            prefix = current + letter
            index = bisect_left(self.sorted_words, prefix)
            if index < len(self.sorted_words) and self.sorted_words[index].startswith(prefix):
                choices.append(ord(letter))
                if letter != "'":
                    choices.append(ord(letter.upper()))
        return choices

    def finish(self, tokens, initial_length):
        current = self.current(tokens)
        if current and current not in self.words:
            return tokens[:max(initial_length, len(tokens) - len(current))]
        return tokens


@torch.inference_mode()
def complete(root, prompt, seed, max_bytes, use_dictionary=True, checkpoint=None):
    torch.set_num_threads(1)
    meta = stacked.read_meta(root)
    if 'crawl' in meta:
        use_dictionary = False
        if checkpoint is None:
            latest = root / f'stage-{len(meta["blocks"]):03d}' / 'latest.pt'
            checkpoint = latest if latest.is_file() else None
    if not meta['blocks'] and checkpoint is None:
        raise ValueError('Train a candidate before testing a prompt.')
    context = meta['config']['context']
    initial = prompt.encode('utf-8')
    if not prompt.strip() or len(initial) > context:
        raise ValueError(f'Enter a nonblank prompt of at most {context} UTF-8 bytes.')

    stream = 'crawl' in meta and bool(meta['blocks'])
    model, depth = stacked.build_model(root, meta, with_frozen=not stream)
    if checkpoint is not None and not stream:
        model.load_state_dict(stacked.load(checkpoint)['trainable'], strict=False)
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    tokens = bytearray(initial)
    lexicon = EnglishLexicon(prompt) if use_dictionary else None
    for _ in range(max_bytes):
        inputs = torch.tensor([list(tokens[-context:])], dtype=torch.long)
        if stream:
            length = inputs.shape[1]
            mask = torch.ones(length, length, dtype=torch.bool).triu(1)
            values = model.token(inputs) + model.position(torch.arange(length))
            for block in meta['blocks']:
                model.block.load_state_dict(stacked.load(root / block['file']))
                values = model.block(values, mask)
            if checkpoint is not None:
                model.load_state_dict(stacked.load(checkpoint)['trainable'], strict=False)
                values = model.block(values, mask)
            logits = F.linear(model.norm(values[:, -1]), model.token.weight)
        elif checkpoint is None:
            logits = F.linear(model.norm(model.bottom(inputs)[:, -1]), model.token.weight)
        else:
            logits = model(inputs)[:, -1]
        if lexicon is not None:
            allowed = list(lexicon.choices(tokens))
            restricted = torch.full_like(logits, -torch.inf)
            restricted[0, allowed] = logits[0, allowed]
            logits = restricted
        tokens.append(torch.multinomial((logits[0] / 0.8).softmax(-1), 1, generator=generator).item())
    if lexicon is not None:
        tokens = lexicon.finish(tokens, len(initial))
    response = tokens[len(initial):].decode('utf-8', errors='replace')
    response = response.replace('\r\n', '\n').replace('\r', '\n')
    response = ''.join(c if c.isprintable() or c in '\n\t' else '\ufffd' for c in response)
    return {'blocks': depth, 'seed': seed, 'response': response}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-bytes', type=int, default=120)
    parser.add_argument('--raw', action='store_true', help='sample without the English dictionary')
    args = parser.parse_args()
    if not 0 <= args.seed < 2**64 or not 1 <= args.max_bytes <= 256:
        parser.error('Use a seed from 0 to 2^64−1 and a generation length from 1 to 256 bytes.')
    try:
        print(json.dumps(complete(args.stack, sys.stdin.buffer.read().decode('utf-8'), args.seed, args.max_bytes, not args.raw)))
    except (OSError, ValueError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
