# M2-B completion report — portable `wgpu` rendering foundation

Scope: rendering only. No Hartree-Fock, no DFT, no molecular chemistry, no CUDA, no
interactive viewer. No GPU compute kernel exists in this milestone, and no
scientific speed-up is claimed anywhere.

Artefacts referenced below are committed under
`docs/validation/m2b-artifacts/` and can be regenerated with a single command.

---

## 1. Commit and working-tree state

| item | value |
|---|---|
| implementation commit | **`c4ff579`** — "M2-B: portable wgpu rendering foundation" |
| previous milestone commit | `421c19f` (M2-A) |
| working tree | clean (`git status --short` empty) after the implementation commit |
| feature flags needed for the GPU path | `atoms-render/gpu` only |

The GPU path is behind an off-by-default feature, so the default workspace build
needs no graphics stack. `cargo test --workspace` neither compiles nor requires it.

---

## 2. Files changed

### New

| file | contents |
|---|---|
| `crates/atoms-render/src/volume.rs` | `VolumeData`, `SliceAxis`, `TransferFunction` — the neutral interface |
| `crates/atoms-render/src/image.rs` | `RgbaImage`, `CompareReport`, viridis map, dependency-free PNG encoder |
| `crates/atoms-render/src/cpu.rs` | CPU reference implementation of every render (always compiled) |
| `crates/atoms-render/src/error.rs` | `RenderError` and its classification |
| `crates/atoms-render/src/gpu/mod.rs` | `wgpu` context, `BackendInfo`, capability checks, buffer readback |
| `crates/atoms-render/src/gpu/slice.rs` | `SliceRenderer`: upload, slice, MIP, readback |
| `crates/atoms-render/src/gpu/slice.wgsl` | WGSL vertex + fragment shaders |
| `crates/atoms-render/tests/orbital_volume.rs` | integration test on a real 2p_z density |
| `crates/atoms-render/examples/orbital_slice.rs` | runs the deterministic scene and writes PNG + metadata |
| `docs/rendering.md` | rendering architecture, interface contract, backends, limits |
| `docs/validation/m2b-artifacts/` | the rendered images, the agreement table, `scene.json` |

### Modified

| file | change |
|---|---|
| `crates/atoms-render/Cargo.toml` | `gpu` feature (`wgpu` 30 + `bytemuck` + `pollster`); no default dependencies |
| `crates/atoms-render/src/lib.rs` | replaced the M2-B stub with the real crate documentation |
| `Cargo.toml` | workspace `wgpu` pinned to a single version (`0.20` → `30`) |
| `crates/atoms-gpu/Cargo.toml` | inherits the workspace `wgpu` instead of pinning its own |
| `crates/atoms-core/src/spatial_grid.rs` | added `public_linear_index`, the single definition of the public array order |
| `crates/atoms-core/src/lib.rs` | re-exports `public_linear_index` |
| `crates/atoms-python/src/ordering.rs` | `numpy_linear_index` delegates to `atoms_core::public_linear_index` |
| `tests/validation/tests/integration.rs` | `atoms-render` moved from the stub list to an *implemented* list that must not claim to be a stub |
| `crates/atoms-python/python/atoms/__init__.py` | `atoms.viewer` **removed** (see §8, decision 4) |
| `python/tests/test_atoms.py` | asserts the viewer is absent; adds explicit seed-determinism tests |
| `docs/python-api.md`, `README.md` | viewer status, M2-B status, rendering pointer |

---

## 3. Architecture decisions

**ADR-1 — the renderer depends on no physics crate.** `atoms-render`'s default
build has *no dependencies at all*. The two sides meet at `VolumeData`, a plain
description of a grid and its values. The bridge is five lines and lives in the
caller. The dependency direction is fixed in documentation: the viewer and any
bridge code depend on both sides; `atoms-core` and `atoms-orbitals` never depend on
the renderer.

*Consequence:* the physics core stays usable on a machine with no GPU and no
graphics stack, and the renderer can be replaced without touching the physics.

**ADR-2 — one definition of the public array order.** The layout
`(i * ny + j) * nz + k` now exists once, as `atoms_core::public_linear_index`.
`SpatialGrid3D::index_public` and the Python binding's ordering module both call it,
so the NumPy arrays Python returns and the volumes the renderer consumes cannot
drift apart. `atoms-render` re-implements the formula because it may not depend on
`atoms-core`; its layout is pinned independently by an asymmetric-grid test whose
values encode their own coordinates, and the orbital integration test confirms the
two agree on real data.

**ADR-3 — the scalar intensity travels in the alpha channel.** RGB carries the
display colour, alpha carries the quantised scalar with no colour map. Validation
compares alpha only, so a colour-map difference cannot be mistaken for a physics
difference.

**ADR-4 — the render target is `Rgba8Unorm`, not `-Srgb`.** No automatic
transfer-function encoding is applied, so the bytes in memory are exactly what the
shader wrote and the comparison with the CPU needs no inverse transform.

**ADR-5 — the volume travels as a storage buffer, not a 3-D texture.** A storage
buffer has one owner, is trivially read back for verification, and avoids
filtering-capability questions entirely. Slice planes coincide with voxel centres,
so nearest-neighbour access is exact here. A `texture_3d` path is the right
evolution once interpolation is needed; it is deferred to M2-C.

**ADR-6 — reproducibility is built in, not bolted on.** Rounding uses
`floor(x + 0.5)` on both sides because WGSL's `round` has implementation-defined
tie behaviour. The uniform struct is `#[repr(C)]` with a test asserting the exact
byte offsets the WGSL layout expects.

**ADR-7 — `wgpu` has one version in the graph.** The workspace pinned `wgpu` 0.20
while the renderer needed 30; the workspace entry was updated and `atoms-gpu` now
inherits it, so two generations of the same library cannot end up linked.

---

## 4. Commands executed and their results

```bash
cargo test --workspace                                   # 213+ passed, 0 failed
cargo test -p atoms-render                               # CPU-only path
cargo test -p atoms-render --features gpu                 # GPU path included
cargo clippy --workspace --all-targets --features atoms-render/gpu   # 0 warnings
cargo fmt --all --check                                  # clean
.venv/bin/maturin develop --release                       # extension rebuilt
.venv/bin/python -m pytest python/tests -q                 # Python suite
cargo run -p atoms-render --features gpu --example orbital_slice -- target/orbital-artifacts
```

Exact counts are recorded in §5.

---

## 5. Test results

All commands were run in a single sweep; every step exited 0. Log:
`target/m2b-final.log` and `target/m2b-release.log` (not committed — both are
reproducible with the commands in §4).

| command | result |
|---|---|
| `cargo fmt --all --check` | clean (exit 0) |
| `cargo clippy --workspace --all-targets --features atoms-render/gpu` | **0 warnings** |
| `cargo test --workspace` | **234 passed, 0 failed** |
| `cargo test -p atoms-render` | CPU path only, passed |
| `cargo test -p atoms-render --features gpu` | **69 passed, 0 failed** |
| `cargo test -p atoms-core` | passed |
| `cargo test -p atoms-python` | passed |
| `maturin develop --release` | wheel `atoms-0.1.0-cp315-cp315-macosx_11_0_arm64.whl`, installed |
| `pytest python/tests -q` | **54 passed** |

The 69 tests of the GPU-featured rendering crate are:

* 57 unit tests — layout, transfer function, image/PNG, error classification,
  capability checks, and the GPU renderers against the CPU reference;
* 11 integration tests on a real 2p_z density (one of them GPU-only);
* 1 doc test.

**M1 regression status: intact.** The workspace total rose from 183 (M2-A) to 234;
11 of the difference are M2-B's, and no M1 test was removed, weakened, or deleted.
Two M1-era tests were *changed*, both correctly:

* `unimplemented_crates_declare_their_status` listed `atoms-render` as a stub. It now
  lists it among the *implemented* crates and asserts that an implemented crate must
  **not** claim to be a stub — so the test got stronger, not weaker.
* The pointer to `atoms.viewer` in the Python suite became an assertion that the
  name is **absent** (see §8).

On a machine without a GPU the GPU tests print a reason and return early rather than
failing; the capability-policy tests run everywhere because they are pure functions.

---

## 6. Scientific validation

The rendering claim is "the GPU produces the same image as the reference
implementation". Four independent lines of evidence were produced.

### 6.1 Upload integrity — bit-identical

The volume buffer is read back from the device and compared against the CPU array by
ULP difference. Measured: **max ULP difference = 0**. The shader sees exactly the
bytes the CPU uploaded; no arithmetic is involved in the transfer.

### 6.2 Rendered images — byte-identical

The committed artefacts let this be checked without rebuilding:

| image | CPU SHA-256 | GPU SHA-256 | result |
|---|---|---|---|
| MIP along y | `3398f128…f1c25` | `3398f128…f1c25` | **identical** |
| slice z = +2 bohr | `4deff880…e2cef` | `4deff880…e2cef` | **identical** |
| nodal plane z = 0 | `732856c5…5e4e2a` | `732856c5…5e4e2a` | **identical** |

Per-pixel, on 19881 pixels each: max |Δ| = 0, mean |Δ| = 0, differing pixels = 0.

The declared tolerance is **1 quantisation step**, and the reason it is not declared
as 0 is honest: the two sides evaluate `ln` with different implementations, so a
value sitting exactly on a quantisation boundary could flip a byte. On this data it
does not, at any of the three images.

### 6.3 Physics anchors verified through the rendered output

The scene is the 2p_z orbital, for which `|psi|^2 ∝ z^2 e^{-r}` is known in closed
form:

| assertion | result |
|---|---|
| the `z = 0` nodal plane renders as an entirely black image | 0 of 19881 pixels non-zero, CPU **and** GPU |
| the brightest pixel of the peak slice is the point on the z axis | holds |
| the global maximum of the field sits on the z axis at \|z\| = 2 bohr | holds (grid point at exactly 2.0 bohr) |
| the projection is mirror-symmetric through z = 0 | holds, asserted row by row |
| both lobes are bright in the projection | holds |
| the field is non-negative everywhere | enforced at construction (`require_non_negative`) |
| integrated probability over the grid | 0.997981 (finite-grid truncation, as expected) |

The rendered images were also **inspected**: the MIP shows two lobes separated by a
dark nodal band; the slice through the maximum shows a single centred blob fading
radially. Both are what the physics requires.

### 6.4 A discovery worth recording

The density on the nodal plane is not *exactly* zero: the strongest value found was
**2.019e-35** against a peak of **5.385e-3**, a ratio of **3.75e-33**.

This is not a physics error and not a bug in M1. It is a floating-point residue from
the coordinate round-trip: a Cartesian point is converted to spherical coordinates,
and the angular factor is `cos(theta)` with `theta = atan2(rho, z) = pi/2` exactly,
where `cos(pi/2)` is `6.1e-17` in f64 rather than 0. Squared, it lands at 1e-35.

The consequence is recorded rather than hidden, and it is harmless: it is 33 orders
of magnitude below the peak, so it quantises to byte 0 and the rendered nodal plane
is exactly black. The test therefore asserts the residue relative to the peak (a
scale-invariant statement) and separately asserts that the *rendered* plane is
black — a claim that holds exactly.

A related measurement artefact was found in the same place: the naive *per-value*
relative f32 conversion error is 1.166e-6, well above `2^-24`, purely because the
nodal-plane values are **subnormal** in f32, where relative precision is poor by
construction. The metric that matters for rendering is the error relative to the
field maximum, and that is **4.299e-8**, inside `2^-24`. Both metrics are now
computed, documented, printed, and asserted, because reporting only the first would
have suggested a precision problem that does not exist.

### 6.5 Transposition detection

Every layout test uses a volume that is asymmetric on all three axes with values
encoding their own coordinates (11×7×5, and 3×5×7 in the core). A symmetric volume
would pass with a transposed index map; these cannot.

---

## 7. Rendering evidence

Command:

```bash
cargo run -p atoms-render --features gpu --example orbital_slice -- target/orbital-artifacts
```

Output (verbatim, abridged to the measurements):

```text
orbital                 2p_z (n=2, l=1, m=0), hydrogen, atomic units
grid                    141^3 over [-12, 12] bohr, spacing 0.171429 bohr
points                  2803221
volume bytes (f32)      11212884
f32 error rel. to max   4.299e-8  (bound 6e-8 = 2^-24)
max density             5.380508e-3 bohr^-3
integrated probability  0.997981

backend:            metal
adapter:            Apple M4 Max
device type:        integrated-gpu
max texture 2d/3d:  8192/2048
max buffer:         268435456 bytes
max storage binding:134217728 bytes

nodal plane              0 of 19881 pixels non-zero (expected 0)
peak slice max byte      255 of 255
upload integrity         bit-identical (max ULP = 0)
```

### Measured timings

Release build (`lto = true`, `codegen-units = 1`), Apple M4 Max, Metal, 141³ volume
(11.2 MB), three 141×141 images. Three independent runs:

| operation | run 1 | run 2 | run 3 | debug build |
|---|---|---|---|---|
| field evaluation, 2.8 M points (CPU) | 78.0 ms | 88.9 ms | 104.7 ms | 561 ms |
| f64 → `VolumeData` conversion | 5.92 ms | 4.10 ms | 4.38 ms | 45.7 ms |
| **three images on the GPU** (upload excluded) | 24.9 ms | 21.3 ms | 27.7 ms | 18.0 ms |
| **three images on the CPU** (single-threaded) | 7.5 ms | 6.2 ms | 8.7 ms | 96.1 ms |
| volume upload to device (11.2 MB) | 1.83 ms | 2.10 ms | 1.87 ms | 1.54 ms |

### The GPU path is *slower* than the CPU path here, and that is reported, not hidden

In release, rendering three images on the GPU costs 21–28 ms against 6–9 ms on a
single CPU thread: **the CPU is roughly 3–4× faster.** This is the headline
performance result of the milestone, and it is negative.

It is also not surprising once the measurement is decomposed. Each GPU image pays for
render-target creation, pipeline binding, command encoding, a submission, a staging
buffer allocation, a texture-to-buffer copy, a device poll and a map — a synchronised
round trip. The CPU number is pixel writes into a `Vec<u8>`:

| component | measured |
|---|---|
| upload of 11.2 MB, once | ~1.9 ms |
| GPU: create + submit + copy + poll + map, per image | ~7 ms |
| CPU: quantise + colour, per image | ~2 ms |

Three consequences, all of which shape M2-C:

1. **This offscreen-plus-readback path is a validation harness, not a performance
   path.** Its purpose is to make the GPU's output comparable to a reference, and
   that requires the pixels on the host. An interactive viewer never reads back: it
   renders into a surface and presents. Once readback is gone, the comparison above
   stops being the relevant one.
2. **Per-call overhead dominates at this size, not throughput.** A 141² image is
   20 k pixels. Amortising means keeping the volume resident on the device (which
   `upload_volume` + `render_with_buffer` already allow) and reusing a persistent
   render target.
3. **The real cost is the field evaluation** — 78–105 ms against 6–9 ms for all the
   rendering. Computing the density in the first place is where GPU compute could
   actually matter, and that is M5's question, not M2-B's.

No GPU acceleration is claimed, and no GPU-vs-CPU speed-up in scientific computation
is claimed, because **no GPU compute kernel exists in this milestone**.

---

## 8. Python API corrections requested in the M2-A acceptance

| # | requested | done |
|---|---|---|
| 1 | units policy documented, derived quantities included | `docs/python-api.md`; density unit stated as `bohr^-3` and carried in `VolumeData::value_unit` |
| 2 | axis order tested with an asymmetric grid with distinct dimensions **and** values | three such tests: `atoms-core::spatial_grid` (3×5×7), `atoms-python::ordering` (17×33×9), `atoms-render` (11×7×5); each value encodes its own coordinates and every axis and index is checked |
| 3 | profile API accepts scalar or 1-D array, rejects lists, tested and documented | unchanged and still covered |
| 4 | do not expose a viewer stub | **`atoms.viewer` removed entirely.** There was no compatibility requirement that justified a placeholder, so the strictest reading was taken: the name does not exist, `test_no_viewer_is_exposed_before_it_exists` asserts its absence, and the docs describe it as M2-C work rather than as a function |
| 5 | sampling: identical seeds give identical output, different seeds are not ignored | added explicit tests on both `sample_radial` and `sample_positions`; different seeds must *differ*, with no claim that every sample is distinct |

---

## 9. Backend status (from actual execution)

| backend | status | basis |
|---|---|---|
| **Metal** | **tested** | Apple M4 Max, macOS 26.7.1, adapters selected via `wgpu` 30; 2.8 M-point volume uploaded, three images rendered, read back and compared |
| Vulkan | **untested** | no Vulkan-capable device on this machine; `wgpu` selects Metal on macOS |
| MoltenVK | **untested** | not installed, not evaluated |
| DX12 | **untested** | no Windows machine available |
| GL | **untested** | not attempted |
| native Vulkan (`ash`) | **not implemented** | deferred until a portable baseline and profiling justify it |
| CUDA | **not implemented** | deferred; `atoms-gpu` keeps the seam |

Compilation is not evidence. Only Metal has been exercised, and the claim is limited
to what was run. Verifying Vulkan requires a Linux or Windows machine with a
Vulkan-capable device; that is a scoped follow-up rather than something this
milestone can complete.

Capability handling was exercised on the negative side too: requiring an impossible
limit returns `ResourceLimit` rather than panicking, and requiring a feature the
adapter lacks returns `UnsupportedCapability`. Both policies are pure functions and
are unit-tested where no GPU is needed, so they hold on machines that cannot run the
rest of these tests.

---

## 10. Performance claims

None beyond the debug-build wall-clock measurements in §7, which are labelled as
such. Specifically:

* no GPU compute speed-up is claimed, because there is no GPU compute;
* no release-build comparison is claimed;
* the CPU reference is single-threaded by design, so its cost is a reference cost,
  not a best-case CPU cost.

---

## 11. Remaining limitations

1. **Slices and axis-aligned projections only.** No camera, no arbitrary view
   direction, no interpolation. Both are M2-C.
2. **No volume rendering** (ray marching, isosurfaces, gradients).
3. **No 3-D texture path**, for the reasons in ADR-5.
4. **No timestamp queries or GPU profiling counters**; timings are wall-clock.
5. **Vulkan and every non-Metal backend remain unverified.**
6. **The PNG encoder does not compress** — files are ~5× larger than a compressing
   encoder's.
7. **No window, no surface, no swap chain.** Rendering is offscreen and read back to
   the host. A presentable surface arrives with the M2-C viewer.
8. **The single-threaded CPU reference is not the fastest CPU path**; it is the
   correctness path.
9. **`atoms-render`'s examples and tests depend on the physics crates** (as Cargo
   `dev-dependencies` only). The library's own dependency graph is free of physics
   coupling, and the distinction is documented rather than glossed over.

---

## 12. Next milestone

**M2-C — interactive orbital explorer**, as specified: quantum-number selection,
interactive 3-D camera, electron-density visualisation and/or isosurfaces, adjustable
thresholds, correct handling of complex wavefunctions, background computation with
cancellation, and visible units/model/error reporting. Not started.

Three things from M2-B carry directly into it:

* the `VolumeData` interface and its ownership rules are already fixed;
* trilinear sampling and a `texture_3d` path become necessary as soon as the view
  direction is arbitrary — the seam exists in `SliceRenderer`;
* `SliceRenderer::upload_volume` + `render_with_buffer` already separate upload from
  render cost, which is what a real-time loop needs.
