(benchmarks)=
# Benchmarks

These empirical reference runs measure one fixed workflow, dataset, software
revision, and cloud resource envelope. The largest completed run processed
10 million input cells through conversion, quality control, normalization,
graph construction, embedding, clustering, and marker search.

The results establish execution and resource use for this configuration. They
are not hardware guarantees, biological validation, or a comparison with
another package.

## 1. End-to-end results

The table reports end-to-end wall time and sampled peak memory for each input
size.

| Input cells | CPU | Container | n | Wall time | Peak memory |
| ----------: | --: | --------: | -: | --------: | ----------: |
| 10,000 | 4 | 16 GiB | 2 | 14.2 ± 2.9 min | 2.6 GiB |
| 50,000 | 4 | 16 GiB | 2 | 15.5 ± 8.3 min | 5.2 GiB |
| 100,000 | 4 | 16 GiB | 2 | 18.5 ± 2.8 min | 6.9 GiB |
| 500,000 | 8 | 32 GiB | 2 | 30.6 ± 5.2 min | 26.8 GiB |
| 1,000,000 | 8 | 32 GiB | 2 | 52.6 ± 10.5 min | 28.9 GiB |
| 5,000,000 | 16 | 64 GiB | 2 | 3.23 ± 0.52 h | 57.2 GiB |
| 10,000,000 | 16 | 64 GiB | 1 | 7.19 h | 56.4 GiB |

`±` is the sample standard deviation. Peak memory is sampled
`memory.current`. Exact seconds, observed ranges, and individual replicate
totals are in the
[profiling record](https://github.com/NygenAnalytics/scarf/blob/master/profiling/BENCHMARKS.md).

(umap-gallery)=
## 2. UMAPs across scale

These embeddings show the saved output from three reference runs, colored by
CELLxGENE development stage.

::::{container} benchmark-gallery
:::{figure} ../_static/benchmarks/umap_development_stage_100000.png
:alt: UMAP from the 100,000-cell reference run colored from early to late Theiler stage
:width: 100%

**100k input:** 88,955 filtered cells shown
:::
:::{figure} ../_static/benchmarks/umap_development_stage_1000000.png
:alt: UMAP from the 1,000,000-cell reference run colored from early to late Theiler stage
:width: 100%

**1M input:** 500,000 of 889,974 filtered cells shown
:::
:::{figure} ../_static/benchmarks/umap_development_stage_10000000.png
:alt: UMAP from the 10,000,000-cell reference run colored from early to late Theiler stage
:width: 100%

**10M input:** 500,000 of 8,902,268 filtered cells shown
:::
::::

:::{image} ../_static/benchmarks/umap_development_stage_legend.png
:alt: Development-stage color key from Theiler stage 12 through Theiler stage 27
:class: benchmark-gallery-legend
:width: 92%
:align: center
:::

Each input size has an independently fitted UMAP, so coordinates are not
aligned across panels. Development stages came from the source CELLxGENE
metadata using the deterministic sample rows, with cell IDs used to verify the
mapping. These panels provide visual context, not biological validation.

(stage-timings)=
## 3. Stage breakdown

Values are elapsed seconds for each stage. Columns from 10k through 5M are
means of two replicates; the 10M column is one completed run. Stage values
exclude orchestration, while dataset download is shown separately.

| Stage | 10k | 50k | 100k | 500k | 1M | 5M | 10M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dataset download | 2.4 | 95.3 | 31.8 | 92.5 | 242.1 | 1,335.4 | 2,174.9 |
| Create count store | 5.6 | 18.0 | 40.4 | 178.8 | 521.7 | 1,310.3 | 3,481.2 |
| Write `countsT` | 8.2 | 19.4 | 40.5 | 100.2 | 190.4 | 619.1 | 1,384.8 |
| Initialize datastore | 30.4 | 29.5 | 40.1 | 63.5 | 99.8 | 369.7 | 683.0 |
| Reopen datastore | 10.1 | 8.8 | 9.7 | 10.5 | 9.2 | 10.2 | 10.3 |
| Filter cells | 30.7 | 25.5 | 31.4 | 30.3 | 30.4 | 41.0 | 54.0 |
| Mark HVGs | 61.0 | 60.4 | 89.3 | 142.9 | 246.1 | 648.7 | 1,366.9 |
| Normalize | 36.6 | 38.5 | 63.2 | 135.7 | 218.2 | 948.5 | 1,296.4 |
| PCA | 43.8 | 35.4 | 49.4 | 70.7 | 101.9 | 264.0 | 570.9 |
| Build embedding initialization | 12.8 | 11.0 | 14.2 | 18.5 | 27.8 | 111.4 | 246.5 |
| Build ANN index | 23.3 | 21.6 | 30.6 | 74.0 | 125.1 | 476.3 | 831.5 |
| Query neighbours | 26.8 | 21.2 | 24.5 | 37.0 | 53.0 | 142.2 | 299.8 |
| Build connectivity map | 53.1 | 55.7 | 51.6 | 52.4 | 55.8 | 64.7 | 101.8 |
| UMAP | 66.0 | 72.9 | 104.8 | 182.9 | 323.7 | 681.0 | 1,772.2 |
| Leiden | 55.6 | 55.5 | 65.3 | 143.1 | 257.6 | 1,380.2 | 3,486.5 |
| Paris | 112.8 | 98.8 | 116.2 | 133.6 | 156.5 | 353.6 | 769.2 |
| Marker search | 115.4 | 118.4 | 139.0 | 181.2 | 282.2 | 2,231.9 | 6,117.9 |

At 10M, marker search was the largest stage; Leiden and initial store creation
were next. At the smallest sizes, fixed work makes the 10k and 50k totals
similar despite the difference in cell count.

(what-was-measured)=
(shared-analysis-settings)=
(machine-classes)=
(how-to-read-these-numbers)=
## 4. Method and limits

| Item | Recorded setting |
| --- | --- |
| Measurement | Completed 2026-08-02 from commit `ba6dc04d7f4e18e441e07d1f503722ef1018f1ff` |
| Source | CELLxGENE dataset `dcfd4feb-18a3-4b30-81d7-1b0c544a8ab3`, version `1bc30289-9565-4099-abf9-3326328c11ac` |
| Sampling | Nested deterministic samples, seed 0 |
| Analysis | 2,000 highly variable features, 50 PCA dimensions, 11 neighbours, 1,000 embedding centroids |
| Graph and clustering | Graph seed 4466; 300 UMAP epochs; UMAP and Leiden seed 4444; Leiden resolution 1.0 |
| Filtering | 1st and 99th cell quantiles; minimum 10 features per cell and 20 cells per feature |
| Execution | Parallel ANN and UMAP on S3-compatible object storage in the Modal EU region |
| Memory planning | Scarf budget set to 75% of each container memory limit |

- Machine size grew with input size. Compare rows only with their recorded
  resource envelope.
- Peak values are sampled, so short memory spikes may be missed.
- Two replicates show run-to-run drift but are insufficient for a useful
  confidence interval. The 10M row has one completed replicate; incomplete
  attempts are excluded.
- These measurements establish execution and resource use for this
  configuration. They do not establish biological correctness or a general
  hardware guarantee.

See {doc}`memory_and_execution` for how `mem_budget` controls planned block
sizes and concurrency. It is not a hard process-memory limit.
