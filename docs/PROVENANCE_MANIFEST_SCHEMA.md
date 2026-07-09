# External Dataset Provenance Manifest

Any `external_benchmark` run requires a JSON manifest bound to the exact dataset file through SHA-256.

```json
{
  "dataset_fingerprint_sha256": "<sha256 of the exact CSV or JSONL file>",
  "source_name": "JudgeBench",
  "source_uri": "<provider URL>",
  "license": "<license string>",
  "revision_or_commit": "<dataset revision, commit, or release identifier>",
  "retrieved_at_utc": "2026-06-26T00:00:00Z",
  "permitted_splits": ["final"],
  "evidence_class": "external_benchmark"
}
```

`evidence_class` must be one of `external_benchmark`, `public_derived_controlled`, `synthetic_fixture`, or `test_only`. The held-out release path accepts only `external_benchmark`. This is an audit and reproducibility control. It does not independently prove a source's license or legal permission; the stated source and license must still be checked against the provider.
