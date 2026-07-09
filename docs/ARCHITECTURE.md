# Architecture

```text
manifest -> schema/split validation -> response pairs -> C0/C1/C2
         -> static or adaptive anchors -> AAL + VCC -> routing
         -> meta-evaluation -> reports/figures/release decision
         -> optional equal-budget DPO transfer experiment
```

The CPU path validates contracts with synthetic data. The optional GPU path is deliberately isolated so a GPU dependency cannot hide data, split, lineage, or routing defects.
