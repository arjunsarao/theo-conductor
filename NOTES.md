# Notes

## General

- Might be worth buying a fugu sub to see which models it uses.

## Sakana Fugu Technical Report

- "a Fugu model constructs and agentic scaffold overe a pool of frontier LLM workers, deciding which workewrs to involve, what instructions or roles to assign, how intermediate  outputs should be combined or verified, and when to synthesize a final answer."
- " A keyt design choice is that Fugu uses the orchestrator's logits rather than its generated text. Since prompting and task execution are delegated to the selected frontier model, the orchestrator only needs to produce a worker-selection decisison" (Is this for Fugu only or also Fugu Ultra?)
- Fugu is trained in 2 stages
  - The first is large-scale SFT on single-step tasks. (I will use MegaScience for this)

Below is the pseudocode for this SFT:

```py
import itertools
import collections
import pprint as pp
import math



def evaluate(model, question, answer) -> tuple[str, float]:
    import random
    return str(random.randint(1, 10)), random.choice([0.0, 0.5, 1.0])


def softmax(z, temperature=1.0):
    return [math.exp(z_i / temperature) / sum(math.exp(z_j / temperature) for z_j in z) for z_i in z]



n = 4 # Number of repititions
D = [("Whats 2+2?", 4), ("Whats 3*4?", 12), ("Whats 4/5?", 0.8)]  # Note the answers are integers, denotes they are verifiable.
M = ["GPT-5.5", "Claude Opus 4.8", "Gemini 3.5 Flash"]


candidate_solutions = collections.defaultdict(list)
for model, (question, answer) in itertools.product(M, D):
    for _ in range(n):
        response, reward = evaluate(model, question, answer)
        candidate_solutions[(model, question)].append(reward)

s = collections.defaultdict(list) # How well each model does for a given question
for (model, question), reward_set in candidate_solutions.items():
    r = sum(candidate_solutions[(model, question)]) / n
    s[question].append(r)


pp.pprint(candidate_solutions)
pp.pprint(s)

# turn this inso a soft target disttrubiotion p_i(j) where i is the question index and j is the model index.
p = collections.defaultdict(list)
for i, (question, answer) in enumerate(D):
    for j, model in enumerate(M):
        p[i].append(softmax(s[question])[j])

pp.pprint(p)

# Then use KL divergence as SFT loss

```

- Then is Section 3.1.3 which is applying evolutionary stragegies on end-to-end tasks.
- Fugu-Ultra is trained only on end-to-end tasks (souce?)
- It uses GRPO where the reward is 0 for invalid parse, 1.0 if the workflow produces the correct answer and 0.5 otherwise.


## Learning to Orchestrate Agents in Natural Language with the Conductor

## TRINITY: An Evolved LLM Coordinator

