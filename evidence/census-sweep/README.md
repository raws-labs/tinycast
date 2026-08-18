# The leaderboard sweep behind the census

One line per GIFT-Eval leaderboard entry at the pinned benchmark commit, with what the
census did with it and why. The paper's census appendix says the sweep covered every
entry and that every entry it did not admit was checked against the criteria; this is
that check, enumerated.

`board-entries.csv` has one row per result directory. `board_model`, `model_type`,
`testdata_leakage`, `org` and `replication_code_available` are the entry's own published
metadata, copied from its `config.json`. `n_configurations`, `nGMASE`, `nCRPS` and
`nMSIS` are recomputed from the entry's own published per-configuration file against the
benchmark's seasonal-naive reference, under the leaderboard's aggregation rule, which is
the same rule `../census/aggregate.py` applies to the 28 entries that ship whole.

`census_disposition` is a rule ladder, first match wins, and `disposition_basis` names
what the match read:

1. no `all_results.csv` at the pinned commit
2. the benchmark's own reference rows, `naive` and `seasonal_naive`
3. `testdata_leakage` is Yes
4. `model_type` is `agentic`: an agentic or ensemble entry rather than a single model
5. `model_type` is `fine-tuned`: fitted to the evaluation series
6. `model_type` is `deep-learning`: the board's category for per-dataset supervised entries
7. `model_type` is `statistical`: fitted per series at inference
8. otherwise a zero-shot candidate, decided on size: admitted, size not establishable, or
   above the 10M cut

Rules 1 to 7 read the pinned metadata and re-derive exactly. Rule 8 does not, and the
table says so per row rather than implying otherwise. `parameters` carries a count only
where one was written down: the eight admitted entries, and ten of the entries above the
cut. For the rest of the entries above the cut, `parameters_source` records that the size
was verified during the sweep and the count was not kept per entry.

The nine directories at "size cannot be established" are seven models; two of them carry
two board directories each.
