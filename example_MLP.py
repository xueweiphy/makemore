"""Train the makemore-2 MLP on names.txt and report against the bigram baseline.

    python example_MLP.py

Set seed = None for an unseeded run.  Everything else is a hyperparameter you
are meant to change -- the point of the file is to make one experiment cheap.
"""

import matplotlib.pyplot as plt

from MLP import MLP


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

block_size = 3           # characters of context
embd_dim   = 4           # embedding dimension per character
h_dim      = 100         # hidden units
minb_size  = 32          # minibatch size
numRun     = 10000       # training steps
lr         = 0.1         # learning rate

seed       = 2147483647  # None for an unseeded run
split_seed = 42          # kept separate: same split, different init

BIGRAM_BASELINE = 2.4546 # from bigram.py, for comparison
UNIFORM         = 3.2958 # log(27), a model that knows nothing


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------

words = open("names.txt", "r").read().splitlines()

model = MLP(words, block_size, embd_dim, h_dim, seed=seed)
model.makeXY()
model.build_data(split_seed=split_seed)
model.init_params()

n_params = sum(p.numel() for p in model.params)
print(f"vocabulary {model.volc_dim}   examples {len(model.Y)}   parameters {n_params}")

model.evo_loss(data="train", numRun=numRun, minb_size=minb_size, lr=lr)


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

train_loss = model.evaluate("train")
dev_loss   = model.evaluate("dev")

print()
print(f"  uniform baseline   {UNIFORM:.4f}")
print(f"  bigram baseline    {BIGRAM_BASELINE:.4f}")
print(f"  MLP train          {train_loss:.4f}")
print(f"  MLP dev            {dev_loss:.4f}")
print()
print(f"  gain over bigram   {BIGRAM_BASELINE - dev_loss:+.4f} nats")
print(f"  perplexity         {2.71828 ** dev_loss:.1f}  (bigram was 11.6)")

# train much below dev means the network is memorising rather than generalising
gap = dev_loss - train_loss
print(f"  train/dev gap      {gap:+.4f}" + ("   <- overfitting" if gap > 0.1 else ""))


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

plt.figure(figsize=(8, 5))
plt.plot(model.lossi, linewidth=0.5, alpha=0.6, label="minibatch loss")
plt.axhline(BIGRAM_BASELINE, color="r", ls="--", label="bigram baseline")
plt.axhline(UNIFORM, color="gray", ls=":", label="uniform baseline")
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"MLP: block_size={block_size}, embd={embd_dim}, hidden={h_dim}, lr={lr}")
plt.legend()
plt.tight_layout()
plt.savefig("mlp_loss.png", dpi=150)
print("\nwrote mlp_loss.png")
