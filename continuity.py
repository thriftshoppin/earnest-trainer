"""Held-out, paired tests for whether a continuation follows its setup."""
import math

import torch
from torch.nn import functional as F


# Each pair changes the cause, evidence or goal. Swapping the endings keeps
# the words and lengths identical across the matched and mismatched scores.
CASES = (
    (('The floor was wet, so Nia went to the closet for a towel.',
      'The room was cold, so Nia went to the closet for a blanket.'),
     (' She used it to dry the floor.', ' She wrapped it around her shoulders.')),
    (('Eli had agreed to meet Jo beside the pool. Jo sent a note saying the pool had closed.',
      'Eli had agreed to meet Jo beside the pool. Jo sent a note saying the pool had reopened.'),
     (' He asked Jo where else they could swim.', ' He took his swimsuit to the pool.')),
    (('Ava put a seed in the soil because she wanted to grow a flower.',
      'Ava put a seed on the feeder because she wanted to feed a bird.'),
     (' She watered the soil.', ' She watched for a bird to land.')),
    (('The sign pointed left for the station and right for the bakery. Omar needed a train.',
      'The sign pointed left for the station and right for the bakery. Omar needed bread.'),
     (' He turned left and looked for the platform.', ' He turned right and looked for the bread shop.')),
    (('June opened the cage because the bird was ready to fly outside.',
      'June opened the cage because the bird needed a fresh bowl of water.'),
     (' She watched the bird fly away.', ' She replaced the empty bowl.')),
    (('Sam expected the box to hold a hat, but the label said it held gloves.',
      'Sam expected the box to hold gloves, but the label said it held a hat.'),
     (' He opened it and tried a glove on his hand.', ' He opened it and put the hat on his head.')),
    (('Max had brought a spoon to eat his soup.',
      'Max had brought a comb to fix his hair.'),
     (' He dipped it into the bowl.', ' He ran it through his hair.')),
    (('The oven timer rang as Rosa finished mixing a second batch. The first batch was done.',
      'The oven timer rang as Rosa finished mixing a second batch. The first batch was still raw.'),
     (' She took the first batch out of the oven.', ' She set the timer for a few more minutes.')),
    (('Dee left the blanket outside to dry. It started raining.',
      'Dee left the blanket outside to dry. The sun came out.'),
     (' She brought the blanket in before it got wetter.', ' She left the blanket in the sun.')),
    (('The dog dropped a ball at Tia\'s feet and waited for her to play.',
      'The dog dropped an empty bowl at Tia\'s feet and waited by the tap.'),
     (' Tia threw the ball across the yard.', ' Tia filled the bowl with water.')),
    (('Ali followed the sound of a piano to find the music room.',
      'Ali followed the smell of fresh bread to find the kitchen.'),
     (' She heard another note from behind the door.', ' She saw a loaf cooling beside the oven.')),
    (('The light was green when Bo reached the crossing.',
      'The light was red when Bo reached the crossing.'),
     (' Bo walked across the street.', ' Bo waited on the sidewalk.')),
)


def score(model, frozen):
    model.eval()

    def loss(prompt, ending):
        prefix = prompt.encode('utf-8')
        suffix = ending.encode('utf-8')
        tokens = prefix + suffix
        if len(tokens) - 1 > model.cfg.context:
            raise ValueError('Continuity test exceeds the model context')
        values = torch.tensor([list(tokens)], dtype=torch.long)
        inputs = values[:, :-1]
        if frozen:
            logits = F.linear(model.norm(model.bottom(inputs)), model.token.weight)
        else:
            logits = model(inputs)
        return F.cross_entropy(logits[0, len(prefix) - 1:], values[0, len(prefix):]).item() / math.log(2)

    margins = []
    with torch.inference_mode():
        for (first, second), (first_end, second_end) in CASES:
            matched = loss(first, first_end) + loss(second, second_end)
            swapped = loss(first, second_end) + loss(second, first_end)
            margins.append((swapped - matched) / 2)
    return {'passes': sum(margin > 0 for margin in margins), 'pairs': len(margins),
            'margin_bpb': sum(margins) / len(margins), 'pair_margins_bpb': margins}


def passes_gate(parent, candidate, general_gain, min_margin, max_regression):
    required_passes = min(parent['passes'] + 1, parent['pairs'])
    return (candidate['passes'] >= required_passes
            and candidate['margin_bpb'] - parent['margin_bpb'] >= min_margin
            and general_gain >= -max_regression)
