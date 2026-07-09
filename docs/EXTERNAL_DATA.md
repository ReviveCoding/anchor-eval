# External Data and License Procedure

Before any external data is used, create a data manifest recording source URL, revision or checksum, license, access date, intended split role, and whether human labels are objective, high-consensus, or ambiguous.

- FinQA: development and deterministic verifier-backed pair generation only.
- JudgeBench: immutable external evaluator test only.
- IFEval: verifiable instruction-following stress test only.
- Human preference artifacts: evaluation-only after source-level provenance and license review.

No final-test item, source group, counterfactual sibling, anchor, or DPO prompt may leak into an incompatible stage.
