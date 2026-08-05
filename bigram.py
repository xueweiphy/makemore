"""Bigram character-level language model on names.txt.

Two ways of fitting the same model:

  1. Counting.  Tally every character pair, add-one smooth, normalise each row.
  2. Gradient descent.  A single 27x27 weight matrix trained on the NLL loss.

They converge to the same thing.  The second is not a better model -- it is the
same model reached by a different route, and that route is the one that
generalises to everything larger.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Data and vocabulary
# ----------------------------------------------------------------------

words = open("names.txt", "r").read().splitlines()

chlist = ["."] + sorted(set("".join(words)))   # '.' is the start/end token, index 0
stoi = {st: i for i, st in enumerate(chlist)}
itos = {i: st for i, st in enumerate(chlist)}
V = len(chlist)


def bigrams(word_list):
    """Yield (context, target) index pairs, with '.' padding at both ends."""
    for ww in word_list:
        ww = "." + ww + "."
        for a, b in zip(ww, ww[1:]):
            yield stoi[a], stoi[b]


# ----------------------------------------------------------------------
# Model 1: counting
# ----------------------------------------------------------------------

NN = torch.zeros((V, V))
for xi, yi in bigrams(words):
    NN[xi, yi] += 1

# Add-one smoothing.  Without it, any pair never observed gets probability 0
# and contributes log(0) = -inf to the loss.
PP = (NN + 1).float()
count_prob = PP / PP.sum(dim=1, keepdim=True)

xs = torch.tensor([xi for xi, _ in bigrams(words)])
ys = torch.tensor([yi for _, yi in bigrams(words)])
N = len(xs)

count_loss = -count_prob[xs, ys].log().mean()
print(f"vocabulary size V   = {V}")
print(f"training examples N = {N}")
print(f"uniform baseline    = {torch.tensor(float(V)).log():.4f}   (perplexity {V})")
print(f"count model  NLL    = {count_loss:.4f}   "
      f"(perplexity {count_loss.exp():.1f})")


def sample(prob, n=5, seed=2147483647):
    """Draw n names by walking the transition matrix from '.' until '.'."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n):
        word, ix = "", 0
        while True:
            ix = torch.multinomial(prob[ix], num_samples=1, generator=g).item()
            if ix == 0:
                break
            word += itos[ix]
        out.append(word)
    return out


print("\nsamples from the count model:")
for w in sample(count_prob):
    print(" ", w)


# ----------------------------------------------------------------------
# Model 2: one linear layer trained by gradient descent
# ----------------------------------------------------------------------

# One-hot encoding makes the matrix product explicit: with X one-hot,
# (X @ W)[a] is just row W[x[a]].  W[xs] is identical and allocates nothing,
# but the one-hot form is written out here to show that an embedding lookup
# *is* a matrix multiply.  Cost of being explicit: N x V floats ~ 25 MB.
g = torch.Generator().manual_seed(2147483647)
xenc = F.one_hot(xs, num_classes=V).float()
W = torch.randn((V, V), generator=g, requires_grad=True)

STEPS, LR, REG = 300, 50.0, 0.01
history = []

for step in range(STEPS):
    # forward
    logits = xenc @ W                        # (N, V)
    counts = logits.exp()
    prob = counts / counts.sum(dim=1, keepdim=True)
    nll = -prob[torch.arange(N), ys].log().mean()
    # L2 on the weights plays exactly the role of add-one smoothing above:
    # it pulls the logits together, which spreads probability toward uniform.
    loss = nll + REG * (W ** 2).mean()
    history.append(nll.item())

    if step % 50 == 0:
        print(f"step {step:4d}   nll {nll.item():.4f}")

    # backward
    W.grad = None
    loss.backward()
    with torch.no_grad():
        W += -LR * W.grad

print(f"step {STEPS:4d}   nll {history[-1]:.4f}")

# The two routes agree.  Gradient descent has rediscovered the counts.
print(f"\ncount model  NLL = {count_loss:.4f}")
print(f"neural model NLL = {history[-1]:.4f}")
print(f"difference       = {abs(count_loss.item() - history[-1]):.4f}")

# Probabilities from W directly: since the input is one-hot, row i of the
# softmax of W is the model's distribution given context character i.
with torch.no_grad():
    wprob = W.exp() / W.exp().sum(dim=1, keepdim=True)

print("\nsamples from the neural model:")
for w in sample(wprob):
    print(" ", w)


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.imshow(NN, cmap="Blues")
for i in range(V):
    for j in range(V):
        ax1.text(j, i, itos[i] + itos[j], ha="center", va="bottom",
                 color="gray", fontsize=5)
        ax1.text(j, i, int(NN[i, j].item()), ha="center", va="top",
                 color="gray", fontsize=5)
ax1.set_title("bigram counts")
ax1.axis("off")

ax2.plot(history)
ax2.axhline(count_loss.item(), color="r", ls="--", label="count model")
ax2.axhline(torch.tensor(float(V)).log().item(), color="gray", ls=":",
            label="uniform baseline")
ax2.set_xlabel("step")
ax2.set_ylabel("NLL")
ax2.set_title("gradient descent converges to the count solution")
ax2.legend()

plt.tight_layout()
plt.savefig("bigram.png", dpi=150)
print("\nwrote bigram.png")


# ----------------------------------------------------------------------
# Note on numerical stability
# ----------------------------------------------------------------------
# The forward pass above computes exp/normalise/log by hand, which is the
# instructive form but not the safe one: logits.exp() overflows once any logit
# exceeds ~88 in float32.  In real code the whole block collapses to
#
#     loss = F.cross_entropy(logits, ys)
#
# which fuses log_softmax with the gather, subtracts the row max internally,
# and has a simpler backward pass: dL/dlogits = (softmax(logits) - onehot)/N.
