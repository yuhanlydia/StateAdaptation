# Oral generalization asset status

Verified on 2026-08-22 UTC.

| Asset | Local path | Status |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | `artifacts/models/Qwen2.5-VL-7B-Instruct` | complete; revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Phi-3.5-Vision-Instruct | `artifacts/models/Phi-3.5-vision-instruct` | complete; revision `12b77fb40b63a2c73c68243d3f767aab688a1b2a` |
| ManipBench upstream code | `data/raw/manipbench-official` | commit `39b4a3c1bd17bcc29e27993f817017040f116e04` |
| ManipBench simplified Q1/Q2 | `data/raw/manipbench-simplified` | 4.26 GB zip downloaded; 52,028 members passed CRC and were extracted |
| Camelyon17-WILDS experiment subset | `data/prepared/camelyon17` | seeds 0/1/2; mirror revision `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f` |
| ManipBench Q1 experiment subsets | `data/prepared/manipbench_q1` | three domains x three seeds complete |

The ManipBench zip SHA-256 is
`0b12f716b692ac7637dcd48ee8ae5f5e94f492eebe9fb57c2957572bcf8edf79`.

Camelyon17's official WILDS CodaLab endpoint returned HTTP 500 during this
preparation. The preparation script therefore uses the public
`wltjr1007/Camelyon17-WILDS` parquet mirror and materializes only registered
support/query rows. Before publication, record and pin the mirror revision and
cross-check sampled metadata against the official WILDS split definition.

Validated split invariants:

- Camelyon17: every seed has support 16 (8/class) and query 300 (150/class).
- Camelyon17: support and query have disjoint sample IDs and patient/slide groups.
- ManipBench: every domain/seed has support 32 (8/option) and query 400
  (100/option), with disjoint sample IDs.
- Domains are `bridge_pick_place`, `droid_pick_place`, and `droid_arti`.
- Repository test suite: 52 tests pass in the preparation environment.

Large assets are intentionally ignored by Git. Recreate them with
`scripts/download_oral_assets.sh`; experiment manifests and asset hashes must
be copied into durable run storage before using an ephemeral GPU node.
