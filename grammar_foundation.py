"""Controlled word-role exercises for Willow's grammar foundation."""
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROLES = {
    'noun': [('The light went out.', 'light'), ('The watch fell.', 'watch'), ('I saw a duck.', 'duck'), ('The paint dried.', 'paint'), ('The brush broke.', 'brush'), ('The train stopped.', 'train'), ('The cook smiled.', 'cook'), ('The seed grew.', 'seed')],
    'verb': [('Light the candle.', 'light'), ('Watch the boat.', 'watch'), ('Duck under the rope.', 'duck'), ('Paint the wall.', 'paint'), ('Brush the dog.', 'brush'), ('Train the puppy.', 'train'), ('Cook the rice.', 'cook'), ('Seed the lawn.', 'seed')],
    'adjective': [('The light bag fell.', 'light'), ('A clean cup stood nearby.', 'clean'), ('He took the long road.', 'long'), ('A quiet room helped.', 'quiet'), ('The red kite flew.', 'red'), ('The sharp knife cut well.', 'sharp'), ('She wore a warm coat.', 'warm'), ('The fast train arrived.', 'fast')],
    'adverb': [('The train moved fast.', 'fast'), ('He spoke softly.', 'softly'), ('She walked slowly.', 'slowly'), ('The bird flew high.', 'high'), ('He arrived early.', 'early'), ('They waited outside.', 'outside'), ('She answered clearly.', 'clearly'), ('He worked hard.', 'hard')],
    'article': [('The gate closed.', 'The'), ('A mouse ran.', 'A'), ('An owl called.', 'An'), ('We saw the moon.', 'the'), ('She took a cup.', 'a'), ('He ate an apple.', 'an'), ('I found the key.', 'the'), ('They saw a bird.', 'a')],
    'pronoun': [('She opened the gate.', 'She'), ('He saw a mouse.', 'He'), ('It flew away.', 'It'), ('They saw the moon.', 'They'), ('We took a cup.', 'We'), ('I ate an apple.', 'I'), ('You found the key.', 'You'), ('Mia helped him.', 'him')],
}
HELD = {
    'noun': [('The bark felt rough.', 'bark'), ('Her ring was gold.', 'ring'), ('A seal swam past.', 'seal'), ('The last spring broke.', 'spring')],
    'verb': [('Dogs bark at night.', 'bark'), ('Ring the bell twice.', 'ring'), ('Seal this envelope.', 'seal'), ('The cats spring up.', 'spring')],
    'adjective': [('They took the right path.', 'right'), ('The early bus was late.', 'early'), ('A hard shell cracked.', 'hard'), ('The high wall blocked us.', 'high')],
    'adverb': [('Turn right at the bridge.', 'right'), ('She woke early today.', 'early'), ('I tried hard again.', 'hard'), ('The balloon rose high.', 'high')],
    'article': [('The lamp flickered again.', 'The'), ('Rita fed a horse.', 'a'), ('An eagle circled above.', 'An'), ('Dee held the old map.', 'the')],
    'pronoun': [('Lena gave her a note.', 'her'), ('The storm surprised us.', 'us'), ('She showed them the shed.', 'them'), ('It fell behind the box.', 'It')],
}


def prompt(sentence, word):
    return f'Sentence: {sentence}\nWord: {word}\nPart of speech:'


def install(root):
    import feedback
    for role, examples in ROLES.items():
        for sentence, word in examples:
            feedback.record(root, {'prompt': prompt(sentence, word), 'response': '',
                                  'blocks': 6, 'seed': 2026}, role + '.', author='grammar curriculum')


def evaluate(root, checkpoint=None):
    import torch
    import torch.nn.functional as F
    import stacked

    torch.set_num_threads(1)
    meta = stacked.read_meta(root)
    model, _ = stacked.build_model(root, meta)
    state = stacked.load(checkpoint) if checkpoint else None
    if state:
        model.load_state_dict(state['trainable'], strict=False)
    model.eval()
    results = []
    with torch.inference_mode():
        for expected, cases in HELD.items():
            for sentence, word in cases:
                prefix = prompt(sentence, word).encode()
                gains = {}
                for role in ROLES:
                    tokens = torch.tensor([list(prefix + f' {role}.'.encode())])
                    inputs = tokens[:, :-1]
                    logits = model(inputs) if state else F.linear(model.norm(model.bottom(inputs)), model.token.weight)
                    gains[role] = F.cross_entropy(logits[0, len(prefix)-1:], tokens[0, len(prefix):]).item()
                results.append({'sentence': sentence, 'word': word, 'expected': expected,
                                'prediction': min(gains, key=gains.get)})
    by_role = {role: sum(row['prediction'] == role and row['expected'] == role for row in results)
               for role in ROLES}
    return {'step': state['step'] if state else 0, 'correct': sum(by_role.values()),
            'total': len(results), 'by_role': by_role, 'examples': results}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('stack', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.stack, args.checkpoint), indent=2))
