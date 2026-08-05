# makemore — bigram character-level language model

A bigram model of names, fitted two different ways, to show they are the same model.

Following Andrej Karpathy's makemore lecture 1, written from memory rather than
copied. Run with `python bigram.py`.

## The model

Predict each character from the one before it. With `.` as the start/end token
the vocabulary is $V = 27$, the corpus gives $N = 228{,}146$ character pairs, and
the model is a single $27 \times 27$ table of conditional probabilities.

## Two routes to the same table

**Counting.** Tally every observed pair into a $27 \times 27$ matrix, add one to
every entry, normalise each row. Done in closed form, no optimisation.

**Gradient descent.** A single weight matrix $W$ of the same shape, trained on
the negative log-likelihood

$$\mathcal{L} = -\frac{1}{N}\sum_{\alpha=1}^{N} \log P_{\alpha, y_\alpha},
\qquad P_{\alpha j} = \frac{\exp Z_{\alpha j}}{\sum_k \exp Z_{\alpha k}},
\qquad Z_{\alpha j} = \sum_i X_{\alpha i} W_{ij}$$

with $\alpha$ indexing training examples and $i,j,k$ indexing characters.

The second route converges to the first. That is the point of the exercise: the
neural network does not learn something cleverer than counting, it *rediscovers*
counting — and unlike counting it keeps working when the context grows beyond one
character.

![bigram counts and training curve](bigram.png)

## Results

| model | NLL | perplexity |
| --- | --- | --- |
| uniform over 27 characters | 3.2958 | 27.0 |
| bigram, counted | 2.4546 | 11.6 |
| bigram, gradient descent | ≈ 2.45 | ≈ 11.6 |

Perplexity reads as an effective number of choices: one character of context
narrows the next character from 27 possibilities to about 12. The loss cannot
reach zero — its floor is the entropy of the names themselves.

## Smoothing is regularisation

Add-one smoothing in the count model and the L2 penalty `0.01 * (W**2).mean()` in
the trained model do the same job. Both pull the distribution toward uniform and
prevent any pair from being assigned zero probability. Remove the L2 term and the
trained model matches the *unsmoothed* counts instead. The two knobs are the same
knob.

## Files

| File | |
| --- | --- |
| `bigram.py` | both models, sampling, figures |
| `names.txt` | 32k names, one per line |
| `bigram.png` | count matrix and training curve (generated) |

## Notes

The forward pass computes `exp` → normalise → `log` explicitly rather than
calling `F.cross_entropy`. That is deliberate — the manual version is what the
derivation looks like — but it is not the form to use in real code: `logits.exp()`
overflows once any logit exceeds about 88 in float32, and the fused version has a
cleaner backward pass, $\partial\mathcal{L}/\partial Z = (P - Y)/N$.

The one-hot encoding is likewise deliberate. `W[xs]` is identical and allocates
nothing, while the one-hot matrix costs about 25 MB; writing it out makes visible
that an embedding lookup *is* a matrix multiplication.

## Next

makemore 2: replace the lookup table with an MLP over character embeddings, so
the context can extend past one character.
