# Data placement

Generated manifests, smoke data, and downloaded archives live here and are intentionally not committed.

- BRIGHT official release: `https://zenodo.org/records/20072020`
- DisasterM3 release: dataset access is provided through the form in `Junjue-Wang/DisasterM3`.
- xBD: download from the official xView2 dataset page after accepting its terms.

Use `scripts/download_bright.sh labels` for the 31 MB instance labels or
`scripts/download_bright.sh full` for the approximately 11.4 GB pre/post/target release.
Some BRIGHT optical scenes require the event-specific official preprocessing tutorial.

The downloaded May 2026 label archive has MD5 `b8b7e8202856966d0f4ff0e3b3aa9b77`.
It contains 3,029 tile-level COCO JSON files and 244,976 train/validation instances.
