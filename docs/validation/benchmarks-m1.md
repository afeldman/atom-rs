# Measured performance — M1

All figures below were produced by one command on one machine:

```text
cargo run --release -p atoms-benchmarks
```

They are wall-clock measurements from that run, not estimates. Reproduce them on
your own hardware rather than quoting these numbers for another machine.

## Environment

| item | value |
|---|---|
| CPU | Apple M4 Max |
| platform | macOS / aarch64 |
| rayon threads | 14 |
| build | release profile, `lto = true`, `codegen-units = 1` |
| repetitions | 7, median reported |
| backend | CPU only |

**No GPU or CUDA figures are reported.** No GPU compute backend exists in M1.

## Results

| benchmark | workload | median | fastest | derived throughput |
|---|---|---|---|---|
| radial `R_nl(r)` | 5 376 evals | 44.08 us | 43.75 us | 122.0 M evals/s |
| spherical harmonic `Y_l^m` | 3 136 evals | 75.83 us | 71.58 us | 41.4 M evals/s |
| density 3D grid (rayon) | 274 625 pts | 1.42 ms | 1.29 ms | 193.7 M pts/s |
| density 3D grid (1 thread) | 274 625 pts | 11.50 ms | 11.33 ms | 23.9 M pts/s |
| density 3D grid (rayon) | 2 146 689 pts | 9.00 ms | 8.15 ms | 238.5 M pts/s |
| density 3D grid (1 thread) | 2 146 689 pts | 93.55 ms | 84.62 ms | 22.9 M pts/s |
| density 3D grid (rayon) | 7 189 057 pts | 35.31 ms | 28.35 ms | 203.6 M pts/s |
| density 3D grid (1 thread) | 7 189 057 pts | 277.75 ms | 241.29 ms | 25.9 M pts/s |
| radial sampling (1 stream) | 1 000 000 samples | 46.45 ms | 45.89 ms | 21.5 M samples/s |
| radial sampling (8 streams) | 1 000 000 samples | 6.22 ms | 6.17 ms | 160.7 M samples/s |

## Measured parallel speedup

Speedup is measured directly against the single-threaded path on the same grid,
not inferred from the thread count:

| grid | points | speedup on 14 threads |
|---|---|---|
| 65^3 | 274 625 | 8.11x |
| 129^3 | 2 146 689 | 10.39x |
| 193^3 | 7 189 057 | 7.87x |

**Interpretation.** Speedup is clearly sub-linear in the thread count. The
plausible causes, in order of likelihood, are (a) the M4 Max has a mix of
performance and efficiency cores, so 14 rayon workers are not 14 equal workers,
and (b) the 65^3 and 193^3 cases are partly dominated by allocation and memory
traffic rather than by arithmetic. The measurements do not separate these
effects, and no claim is made about which dominates.

**What is *not* claimed:** no claim that this scales to other machines, and no
claim of a GPU comparison, because there is no GPU path in M1 to compare against.

## Memory

One density buffer, 8 bytes per `f64` point:

| grid | points | size |
|---|---|---|
| 65^3 | 274 625 | 2.1 MiB |
| 129^3 | 2 146 689 | 16.4 MiB |
| 193^3 | 7 189 057 | 54.8 MiB |
| 257^3 | 16 974 593 | 129.5 MiB |

At 257^3 the buffer alone exceeds 128 MiB. This is the concrete motivation for
the volumetric work deferred to M2/M5, and for considering a narrower element
type for *display* buffers (the scientific buffer stays `f64`).

## Determinism

Parallel density evaluation is asserted to be **bit-for-bit identical** across
runs regardless of thread count (`integration.rs::parallel_density_evaluation_is_bitwise_deterministic`).
Performance measurements therefore do not trade against reproducibility.
