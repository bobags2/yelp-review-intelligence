# Every bug in this project produced a plausible number

*Building a content classification and Trust & Safety pipeline on the Yelp Open Dataset*

Jadon Yeo

---

I spent two years teaching myself machine learning by building trading systems, where a wrong prediction costs money the same day. When I started applying for ML engineering jobs I had no degree and nothing public, so I built something on data anyone can check: the [Yelp Open Dataset](https://www.yelp.com/dataset), 6,990,280 reviews across 150,346 businesses and 1,987,897 users.

This is what came out of it. Seven results, five bugs, and one thread connecting them: not one of those bugs threw an exception. Every single one produced a number I could have put in a table.

---

## 1. The dataset has no fake-review labels, so I didn't build a fake-review classifier

Yelp's filtered reviews aren't published. Any supervised bot detector trained on this data is either using heuristic labels somebody invented — a model learning to reproduce its own rules — or leaking.

So I split the work by what the data honestly supports.

**Supervised head:** multi-label business category classification from review text. The labels are real, sitting in `business.categories`, ~1,300 of them, and the task maps to content type classification.

**Unsupervised head:** behavioural reviewer-anomaly ranking with no labels at all, producing candidates for human review. That's how enforcement actually works. A cheap signal orders a queue, humans work the top of it, and their decisions eventually become the labels a supervised model can train on.

The second framing is less impressive-sounding and much more defensible. I'd rather explain that to a Trust & Safety engineer than explain where my labels came from.

## 2. Hold out businesses, not rows

Reviews name the business they're about. A random row split lets the model memorise "Joe's Diner → Restaurants" from training rows and score inflated on test rows about the same business. A temporal split fails the same way, because a business straddles the cutoff.

I assign whole businesses to train or test by stable hash. On the real data that gave 107,851 training businesses against 27,190 test — 20.1% against a 0.20 target, with zero business overlap by construction.

This is the most common flaw in public notebooks on this dataset, and it inflates results by an amount nobody measures because nobody splits the other way to compare.

## 3. Representation beat model class 93/7, and I ran the counterfactual

First results looked clean. TF-IDF over 200k sparse features with one-vs-rest logistic regression scored **macro AP 0.5680** against a prevalence floor of 0.0484 — 11.7x lift. XGBoost on 256 SVD components plus reviewer metadata scored 0.4190.

Obvious conclusion: gradient boosting is worse at this task.

That conclusion is unsupported, and I nearly shipped it. Two variables changed at once — the model *and* the feature representation. The comparison can't separate them.

The decisive experiment took ten minutes: run logistic regression on the *same* 256-dim feature matrix XGBoost got.

| model | features | macro AP | micro AP |
|---|---|---|---|
| linear | 200k sparse TF-IDF | 0.5680 | 0.5795 |
| linear | 256 dense SVD + metadata | 0.4291 | 0.4932 |
| xgboost | 256 dense SVD + metadata | 0.4190 | 0.4479 |

Logistic regression on XGBoost's features collapses to 0.4291 — about 0.01 from XGBoost, not 0.14 from the sparse linear model. **Representation accounts for ~93% of the gap, model class for ~7%.** Gradient boosting was handed a worse input, not outclassed.

I'd also assumed the low explained variance (0.098 at 256 components) was a defect. It isn't. TF-IDF matrices are near-full-rank and LSA on text routinely captures low variance ratios. Reading it as a bug pointed me at "use more components" when the real question was somewhere else entirely.

## 4. Measuring how lexical the task actually is

If the sparse representation matters that much, the next question is which part of it. I swept chi-squared feature selection at increasing widths, logistic regression at each — same representation type, only the count varying.

| features | macro AP | micro AP | % of peak | fit time |
|---|---|---|---|---|
| 256 | 0.3909 | 0.4287 | 67.8% | |
| 1,000 | 0.4789 | 0.5047 | 83.1% | |
| 5,000 | 0.5516 | 0.5703 | 95.7% | |
| **20,000** | **0.5762** | **0.5945** | **100%** | 224s |
| 100,000 | 0.5735 | 0.5883 | 99.5% | |
| 200,000 | 0.5680 | 0.5795 | 98.6% | 915s |

Three things fell out.

**The signal is concentrated.** 5,000 selected terms recover 95.7% of peak. The task is largely resolved by a few thousand distinctive words — *hygienist*, *cavity*, *rotors*.

**The full vocabulary is actively worse.** 20k beats 200k and fits four times faster. At k=100,000 the weakest selected feature scores chi2 of 3.0, so the last ~180k terms are noise the model works around. My original `--max-features 200000` default was the worst point on the plateau. I'd defend the plateau, not the ranking within it — 0.008 macro AP between 20k and 200k isn't much.

**It reframed the earlier result.** 256 chi2-selected terms score 0.3909, *below* 256 LSA components at 0.4280 (metadata dropped, so the comparison is width-matched at 256 features each). LSA components are linear combinations of all 200k terms, so at equal width the projection carries more information than the 256 best individual terms. What the SVD ablation measured was width, not density.

Per-label, the easy end is rare and lexically distinctive:

| category | prevalence | AP | ROC | lift |
|---|---|---|---|---|
| Dentists | 0.60% | 0.9043 | 0.9914 | 150x |
| Pet Services | 0.78% | 0.7519 | 0.9935 | 96x |
| Fitness & Instruction | 0.98% | 0.8379 | 0.9865 | 86x |

## 5. The transformer gate, and where its margin actually comes from

Now there's a number to beat: **0.5760**, the lexical ceiling.

DistilBERT, two epochs, no tuning, on a matched 60,040-row test set: **0.6335**. The linear model scores 0.5760 on that same 60k against 0.5762 on the full 1.28M, so the sample is representative to 0.0002. The margin is **+0.0575**.

I read that as the residual signal being contextual. That was wrong, and finding out why is the best result in the project.

The alternative explanation is duller: transfer. DistilBERT arrives already knowing *hygienist* relates to dentistry, before it sees a single Yelp review. That's not context disambiguating meaning, it's a better prior on rare terms — exactly where a 20k-term TF-IDF model is thinnest, because it has few examples to estimate those weights from.

The control separates them. Same architecture, same tokeniser, **random initialisation**, no pretrained weights.

| | macro AP | Δ over ceiling | |
|---|---|---|---|
| linear / chi2-20k | 0.5760 | — | lexical ceiling |
| DistilBERT, random init | 0.5903 | +0.0143 | architecture alone |
| DistilBERT, pretrained | 0.6335 | +0.0432 further | plus transfer |

The architecture alone buys 0.0143 over the ceiling. Adding pretrained weights to that same architecture buys a further 0.0432. Those aren't a symmetric split — transfer is measured *conditional* on the architecture, since there's no pretrained-weights-without-the-model counterfactual.

One caveat, stated as a bound rather than chased: under a fixed two-epoch budget, the control was still improving (+0.0225 in epoch 2 against the pretrained model's +0.0092, at higher loss). So this *overstates* transfer's share and understates the architecture's.

Then a second, independent test with a prediction attached before I looked. If transfer is the mechanism, its benefit should concentrate in low-prevalence labels. The architecture's benefit shouldn't.

| delta | slope vs prevalence | r | p |
|---|---|---|---|
| transfer (pretrained − random init) | −0.0312 | −0.405 | **0.0035** |
| architecture (random init − linear) | −0.0008 | −0.025 | 0.86 |

| | rare (<1%) | common (≥10%) |
|---|---|---|
| transfer delta | +0.0781 | +0.0317 |
| architecture delta | +0.0129 | +0.0147 |

Transfer's benefit is 2.5x larger on labels below 1% prevalence. The architecture's is flat across three orders of magnitude.

**The flat arm is the load-bearing one.** The obvious objection to "transfer helps rare labels more" is headroom — rare labels have more room to improve, so any intervention would show that slope. The architecture delta kills it: same 50 labels, same prevalence range (0.65% to 24.5%), same headroom structure, slope −0.0008 at p=0.86. If headroom drove the transfer slope, it would drive this one too. It doesn't.

## 6. What the accuracy costs to serve

Both models exported to ONNX, same box, same provider, same corpus, same batch sizes.

| | peak throughput | ms/review | graph size |
|---|---|---|---|
| TF-IDF + logistic regression | 7,560/s | 0.135 | 10.6 MB |
| DistilBERT | 118/s | 6.9 | 265.7 MB |
| | **64x** | **51x** | **25x** |

64x the serving cost for a 10% relative accuracy gain. That makes the model choice an actual decision rather than a leaderboard result, and the crossover is arithmetic from a per-review compute budget rather than a matter of taste.

Then the model-only number met the pipeline, and the pipeline won.

| consumer batch | throughput | scoring/review |
|---|---|---|
| 2 | 4.8/s | 31.15 ms |
| **8** | **11.4/s** | 44.76 ms |
| 32 | 10.4/s | 65.83 ms |

Two cost curves running in opposite directions. Per-review model cost *rises* with batch size, because batches pad to their longest member and the batch maximum grows with n. Per-batch overhead — offset commit and producing to the output topic — *falls* per review as the batch grows. Their sum minimises near 8.

From the batch-8 run: 2,006 events in 175.5s, of which scoring was 89.8s. That leaves roughly **309 ms per batch** of pure plumbing. So the honest serving claim isn't "our capacity is 11 reviews/s." It's that our capacity is 11 reviews/s and most of it goes to plumbing rather than inference.

Which also means the 64x doesn't survive contact with this pipeline. Put the linear model in the same loop at batch 8 and its ~3.5 ms of scoring disappears into the same 309 ms, giving roughly 26/s — a ~2x realised gap, not 64x. (That figure is arithmetic from two measured numbers, not a run.) Though batch 8 was optimised *for DistilBERT*; re-optimised for a model with almost no padding penalty, the amortisation runs much further and most of the 64x should reassert itself.

## 7. Five bugs, none of which threw

Here's the thread.

**The scaler divergence.** The batch job scored reviewer anomalies with a robust median/MAD scale. Where more than half of accounts sit at the median — which is exactly what happens for `duplication`, since most accounts have zero duplicates — MAD is zero and the code fell back to standard deviation. But it persisted `null` as that signal's scale, and the serving path defaulted to 1.0. Real scale is 0.0532, so the served score understated that component by **18.8x**. An account at the top of the batch queue scored essentially nowhere online. No error, ever.

**The degenerate chi2 target.** A misconfigured selection target made feature selection effectively alphabetical. It reported 256 features "scoring" 0.0524. A number, in a table, wrong.

**The confounded benchmark.** My latency harness cycled a length-varied corpus, so batch size and sequence length grew together. The sweep measured padding, not batching, and confidently recommended batch 2.

**The 10x batching claim.** My own README stated that scoring 32 reviews in one call is "roughly 10x cheaper per review" because fixed per-call overhead dominates. Measured, batch 32 is 2x *worse* than batch 1. That mechanism is real, but its precondition — small model, GPU, or thread-starved — didn't hold for DistilBERT on 8 CPU threads. I asserted it from a general prior and never checked.

**My correction to that claim, which halved throughput.** Having measured that batching costs more per review, I set the consumer default to 2. Throughput dropped to 4.8/s. The benchmark times the model in isolation, in ms per *review*. The commit and produce costs are per *batch*. Two different denominators, and I'd summed them implicitly. The measurement was correct and the conclusion it licensed was wrong.

Two more were latent in the repo rather than mine. `spark.driver.maxResultSize` was never configured, so a 400k-row `toPandas()` aborted at 1,036 MiB against a 1 GB default with 24 GB of driver heap sitting idle. And `PYSPARK_PYTHON` was never pinned, so Spark launched workers with whatever `python3` came first on PATH — which survived a full day of Spark work, because every earlier stage used pure SQL expressions that never start a Python worker.

None of these threw. Every one produced a plausible number. The only thing that caught any of them was asking what the number was actually *of*.

That's also why the counterfactuals matter. The LR-on-SVD ablation and the random-init control aren't extra rigour bolted on — they're the same move applied to modelling claims instead of engineering ones.

And the corrections that stuck are the ones living in tools rather than documentation. `skl2onnx` refuses to convert `strip_accents="unicode"` instead of silently dropping it. ONNX Runtime refuses to build a session without the locale its string normaliser needs, rather than normalising differently from the trained model. My ONNX export verifies numerical parity against PyTorch (3.8e-06 against a 1e-04 tolerance) and exits non-zero otherwise. The parity check confirms 500 sampled accounts match batch scores to 1e-06. And the latency benchmark now prints its own scope, so the next person doesn't repeat my batch-2 mistake.

Guards in commit messages get lost. Guards in tools arrive at the moment of the mistake.

---

## Coda: the queue

The last fix demonstrates the whole thing on one case. The original ranker summed clipped z-scores, and with `duplication`'s median at 0.0043 and its scale at 0.0010, anything above ~0.012 saturated the clip. Two accounts at 0.75 and 0.0156 duplication both scored +8. The signal couldn't tell them apart.

Replacing z-scores with percentile rank through a persisted empirical CDF removes the scale estimate entirely — no MAD, no fallback, nothing to diverge — and both the batch job and the endpoint interpolate the same breakpoint table, so parity is structural rather than agreed. An account with 18 duplicates out of 24 reviews now appears near the top of the queue. It was invisible before, tied with an account at 2 out of 128.

The ranker also weights by expected harm rather than by weirdness. A 3-review burst account affects three reviews; a 200-review account at 0.3 duplication affects sixty. The question isn't "how unusual is this account," it's "which account costs the most if nobody looks."

---

*Code, commits, and every number above: [github.com/bobags2/yelp-review-intelligence](https://github.com/bobags2/yelp-review-intelligence)*
