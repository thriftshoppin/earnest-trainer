"""Approved continuations paired with their original prompts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid


def record(root, draft, approved, author='user'):
    approved = approved.strip()
    if not approved:
        raise ValueError('Enter an approved continuation.')
    prompt = draft['prompt']
    if not prompt[-1].isspace() and approved[0] not in '.,!?;:':
        approved = ' ' + approved
    meta = json.loads((root / 'stack.json').read_text(encoding='utf-8'))
    if len((prompt + approved).encode()) - 1 > meta['config']['context']:
        raise ValueError('Shorten the prompt or continuation to fit the context window.')
    folder = root / 'feedback'
    folder.mkdir(exist_ok=True)
    path = folder / (uuid.uuid4().hex + '.json')
    temporary = path.with_suffix('.tmp')
    item = {**draft, 'approved': approved, 'author': author,
            'action': 'accept' if draft['response'].strip() == approved.strip() else 'correct',
            'saved_at': datetime.now(timezone.utc).isoformat()}
    temporary.write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)
    return path


def examples(root, context):
    folder = root / 'feedback'
    unique = {}
    for path in sorted(folder.glob('*.json'), key=lambda file: file.stat().st_mtime_ns, reverse=True)[:128]:
        item = json.loads(path.read_text(encoding='utf-8'))
        prompt = item['prompt'].encode()
        approved = item['approved'].encode()
        if prompt and approved and len(prompt) + len(approved) - 1 <= context:
            unique.setdefault((prompt, approved), None)
    return list(unique)[:64]
