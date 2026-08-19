# Locked protocol for the new untouched Movie test

This protocol was fixed before collecting mini/nano outputs for the new test reviews.

- Test selection: 200 unique Movie reviews sampled uniformly from reviews missing both mini and nano outputs.
- Selection seed: `20260723`.
- Test manifest: `untouched_200_manifest.json`.
- Models used to establish outcomes: `gpt-5-mini` and `gpt-5-nano`.
- Primary router: the already selected critical-focused, outcome-only direct soft-prompt method.
- Router inputs: operator description and review text only; no semantic gold label or mini/nano answer.
- Training per seed: 200 historical reviews with 133 both-correct, 46 mini-only, 7 nano-only, and 14 both-wrong.
- Validation per seed: the same 40 historical reviews used by the prior experiment.
- Historical 82-row test: ignored; it is neither training nor validation data.
- Seeds: 505, 606, and 707.
- Model: frozen `distilbert-base-uncased`, 4 trainable soft tokens, and a 2-class linear head.
- Training: 8 epochs, batch size 8, learning rate 0.01, max length 192.
- Loss: regret-weighted cross entropy with weights 0.5 (both-correct), 6.0 (mini-only), 1.5 (nano-only), and 1.8 (both-wrong).
- Epoch and threshold selection: old validation data only, maximizing routed answer accuracy and breaking ties toward fewer mini calls.
- Primary reporting: mean and per-seed routed accuracy, gap from all-mini, mini-call savings, mini-only recall, and both-wrong-to-mini rate.
- Test-set tuning, threshold adjustment, and method selection: prohibited.

The untouched labels may be used for a single final evaluation. Any later changes informed by these results require another fresh test set.
