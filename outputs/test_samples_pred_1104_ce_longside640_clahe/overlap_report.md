# Per-tooth overlap report

- images: 4
- detected teeth: 7
- teeth with overlap: 3
- overlapping pairs: 2 (non-adjacent: 0)
- overlapped tooth pixels / total tooth pixels: 0.56%

## By FDI

| FDI | n | overlap>0 (%) | mean ratio (%) | median (%) | p95 (%) | max (%) |
|---|---|---|---|---|---|---|
| 23 | 1 | 100.0 | 0.19 | 0.19 | 0.19 | 0.19 |
| 24 | 1 | 100.0 | 1.63 | 1.63 | 1.63 | 1.63 |
| 25 | 2 | 50.0 | 0.75 | 0.75 | 1.43 | 1.50 |
| 26 | 1 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 27 | 1 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |
| 45 | 1 | 0.0 | 0.00 | 0.00 | 0.00 | 0.00 |

## Top pairs (by frequency)

| pair | adjacent | n images | mean px | max px | mean IoU (%) | mean inter/min (%) |
|---|---|---|---|---|---|---|
| 24-25 | True | 1 | 587 | 587 | 0.77 | 1.54 |
| 23-24 | True | 1 | 31 | 31 | 0.06 | 0.19 |

## Worst teeth (highest overlap ratio)

| file | FDI | area px | overlap px | ratio (%) | neighbors (fdi:px) |
|---|---|---|---|---|---|
| 박애경_20251230_121540.jpg | 24 | 38022 | 618 | 1.63 | 25:587 23:31 |
| 박애경_20251230_121540.jpg | 25 | 39074 | 587 | 1.50 | 24:587 |
| 박애경_20251230_121540.jpg | 23 | 16572 | 31 | 0.19 | 24:31 |
| 박종우B_20250530_165719.jpg | 25 | 41468 | 0 | 0.00 |  |
| 서보석_20250828_160137.jpg | 45 | 25376 | 0 | 0.00 |  |
| 박애경_20251230_121540.jpg | 27 | 30358 | 0 | 0.00 |  |
| 박애경_20251230_121540.jpg | 26 | 28469 | 0 | 0.00 |  |
