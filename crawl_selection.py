"""Validation-based selection for an opt-in crawl stack."""
import json
import math
import random
from pathlib import Path


def accepts(parent_bpb, child_bpb):
    return math.isfinite(child_bpb) and child_bpb <= parent_bpb - 0.001


def publish(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    tmp.replace(path)


def run(root, args, stop_file, trainer):
    import stacked
    import crawl

    meta = stacked.read_meta(root)
    stage = root / f'stage-{len(meta["blocks"]):03d}'
    latest = stage / 'latest.pt'
    parent_file = stage / 'selection-parent.pt'
    if not parent_file.exists():
        parent = stacked.load(latest)
        parent['selection'] = {'bpb': None, 'step': parent['step'], 'rejects': 0}
        stacked.save(parent_file,parent)
    else:
        parent = stacked.load(parent_file)
    step = max(parent['step'],stacked.load(latest)['step'])
    rejected = 0
    parent_samples = parent['selection'].get('samples')
    if parent_samples is None:
        parent_samples = crawl.review(root, 'latest', verbose=False)['samples']
    while stop_file is None or not stop_file.exists():
        # Restore the accepted weights, optimizer moments, and random state before each trial.
        trial = stacked.load(parent_file)
        trial['step'] = step
        if rejected:
            trial['python_rng'] = random.Random(2026+step+rejected).getstate()
            for group in trial['optimizer']['param_groups']:
                group['lr'] *= 0.5**rejected
        lr = trial['optimizer']['param_groups'][0]['lr']
        stacked.save(latest,trial)
        result = trainer(stack=root,steps=args.round_steps,max_minutes=args.round_minutes,
                         batch_size=args.batch_size,threads=args.threads,lr=lr,stop_file=stop_file)
        step = result['step']
        old_score = parent['selection']['bpb']
        if old_score is None:
            old_score = result['initial_validation_bpb']
        score = result['final_validation_bpb']
        interrupted = result['status']=='interrupted'
        accepted = not interrupted and accepts(old_score,score)
        raw_path = stage/'crawl-review-latest.json'
        raw = json.loads(raw_path.read_text(encoding='utf-8'))
        child_samples = raw['samples'] if raw['step']==step and not interrupted else []
        outcome = 'accepted' if accepted else 'stopped; parent restored' if interrupted else 'rejected; parent restored'
        selection = {'outcome':outcome,'parent_bpb':old_score,'candidate_bpb':score,'step':step,'learning_rate':lr}
        publish(stage/'selection-latest.json',selection)
        if child_samples:
            comparison = {**raw,'selection':selection,'samples':[
                *({**sample,'label':'Parent'} for sample in parent_samples),
                *({**sample,'label':f'Candidate ({outcome})'} for sample in child_samples)]}
            publish(stage/'crawl-review-selection.json',comparison)
        if accepted:
            parent = stacked.load(latest)
            parent['selection']={'bpb':score,'step':step,'samples':child_samples}
            stacked.save(parent_file,parent)
            parent_samples = child_samples
            rejected = 0
        else:
            parent['step'] = step
            parent['selection']['bpb'] = old_score
            parent['selection']['samples'] = parent_samples
            stacked.save(parent_file,parent)
            stacked.save(latest,parent)
            rejected += 1
        print(f'Selection: {outcome}. Held-out {old_score:.4f} -> {score:.4f} BPB; learning rate {lr:g}.',flush=True)
        if interrupted or rejected >= 3:
            if rejected >= 3 and not interrupted:
                print('Selection paused after three rejected rounds. Accepted parent saved.',flush=True)
            return
