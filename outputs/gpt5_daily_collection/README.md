# GPT-5 daily Computer Science QA collection

Source: the 410 exactly paired rows in `computer science_result_mini.json` and
`computer science_result_nano.json`.

Models are pinned to:

- `gpt-5-2025-08-07`
- `gpt-5-nano-2025-08-07`

Daily continuation command (run after the 00:00 UTC quota reset):

```powershell
python scripts\collect_computer_qa_gpt5_daily.py --models gpt5 --retry-invalid
```

The collector uses a local 225,000-token UTC-day cap for GPT-5. New rows use a
512-token output cap; cached invalids use 2,048. It checkpoints atomically after
every API response and never re-runs a valid question ID.

GPT-5 nano has already covered all 410 source rows and should not be included in
the recurring command. A small number of exact-prompt abstentions are retained
as `pred: null`, rather than being overwritten with gold labels.

The local ledger cannot observe unrelated API traffic in the same OpenAI quota
group. The safety cap therefore assumes no large concurrent use elsewhere.
