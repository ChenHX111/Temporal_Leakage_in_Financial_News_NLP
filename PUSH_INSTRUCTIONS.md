# PUSH — ready to go (waiting for your repo URL)

The bundle at `EMNLP_REBUTTAL/artifact_bundle/` is a **local git repo with one commit** (`a7423e3`, 87 files, 43.2 MB).
**I have NOT pushed** (per your instruction). When you've logged into the other GitHub account and created an **empty
private** repo, give me the URL and I'll run:

```bash
cd .\Documents\Fin_NLP\EMNLP_REBUTTAL\artifact_bundle
git branch -M main
git remote add origin <YOUR_REPO_URL>            # e.g. https://github.com/<acct>/<repo>.git
git push -u origin main
```
(43 MB total, largest file 23.6 MB < GitHub's 100 MB limit → **no Git LFS needed**.)

## Hosting recommendation (for the corresponding author)
| Host | Fit | Notes |
|---|---|---|
| **HuggingFace Datasets** ★ recommended | best for an NLP dataset | built-in **viewer + streaming loader + DOI**; `HF_DATASET_CARD.md` is ready as the card; 23.6 MB is tiny for HF. |
| GitHub (private now → public later) | good for code+data together | 43 MB fits directly (no LFS); this bundle is already a git repo. |
| Cloud / BigQuery | overkill | only needed for very large data; not necessary at 23.6 MB. |

**Recommended split:** dataset → **HuggingFace Datasets** (card + loader + DOI); code + reproduction + aggregates →
**GitHub**; cross-link them. Both can be private first, flipped public at camera-ready.

## For the REBUTTAL (double-blind) — separate step
The private repo above is for **internal review with the corresponding author** (non-anonymous). For the first-round
rebuttal we must link an **anonymized** artifact — options: `anonymous.4open.science` (mirror the GitHub repo), or an
anonymized HuggingFace repo. I'll prepare the anonymized mirror once the named repo is settled.
