# Static-W8A8 calibration record

What the quantizer's scales were drawn from, and how the series they were drawn from
were chosen. Every exact static-W8A8 number in the paper is computed under these scales.

The appendix states that the 32 calibration series are 16 each from two sources, picked
by ranking each source's series on the SHA-256 digest of its identifier, which reads no
sample value. These files are what make that checkable.

| file | what it holds |
|---|---|
| `selections.jsonl` | one row per selected series: its rank within its source, the digest it was ranked on, and a digest of the exact prepared context |
| `policy.json` | the ranking rule and its preimage, the range estimator, and the aggregate activation ranges the scales were set from |
| `encoder-ranges.txt` | the frozen encoder activation ranges the firmware reads |
| `decoder-ranges.txt` | the frozen decoder activation ranges |

`policy.json` is a reduction. The manifest it came from records absolute paths from the
machine the calibration ran on and internal module names, which are provenance for that
machine and not for a reader, and which cannot be renamed without rewriting a provenance
record. The reduction carries the fields the paper's claims rest on, and the digest of
the manifest it was reduced from, so the chain stays checkable.

Selection is value-independent by construction: the digest is taken over the series
identifier, never over its samples, so no sample value influenced which series were
picked. `policy.json` records that as `selection.value_independent`.
