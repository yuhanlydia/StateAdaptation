---
pretty_name: EventTune-BRIGHT
license: other
---

# EventTune-BRIGHT

Private durable storage for reproducible EventTune dataset derivatives.

Store manifests, event/tile-disjoint splits, checksums, provenance metadata,
and only those derived files whose upstream licenses permit redistribution.
Raw BRIGHT, xBD, and DisasterM3 data are not mirrored here by default.

The prepared Hawaii source gate is stored at
`splits/diagnostics/hawaii-source-event-label-150.jsonl`. It contains 150
examples, balanced across ten source events and three labels, and has SHA-256
`331c5d37b731932322accd78d17e0d72a5fa30bcc29fbe3add427d41812f33de`.
EventTune excludes these sample IDs from source training and evaluates them
before any target-query pass.

Acquisition URLs, expected checksums, and preparation commands live in the
GitHub repository [`Yunbo-max/EventTune`](https://github.com/Yunbo-max/EventTune),
in `data/README.md` and the project README.
