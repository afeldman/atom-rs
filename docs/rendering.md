# Rendering architecture (M2-B)

Status: **M2-B, GPU rendering foundation.** Rendering only. No GPU compute kernel
exists in this milestone, so no scientific speed-up is claimed or implied.

## 1. What this milestone delivers

| capability | state |
|---|---|
| Portable `wgpu` context with typed failures | implemented, tested on Metal |
| Neutral volume interface (`VolumeData`) | implemented, tested |
| Volume upload to a device storage buffer | implemented, bit-identical upload verified |
| Slice rendering along any axis, any index | implemented, validated against CPU |
| Maximum-intensity projection along any axis | implemented, validated against CPU |
| Readback of rendered images to the host | implemented |
| CPU reference implementation of both renders | implemented, always compiled |
| Minimal dependency-free PNG output | implemented, decodes with macOS `sips` and zlib |
| Interactive viewer, camera, isosurfaces | **not in this milestone** (M2-C) |
| GPU scientific compute kernels | **not in this milestone** (M5) |
| CUDA backend | **not in this milestone** (deferred) |

## 2. Layering

```
atoms-core / atoms-orbitals  (physics, CPU, no GPU dependency)
              │
              │  bridge: five lines in the caller (see §3)
              ▼
        VolumeData  ← the only coupling between the two sides
              │
        atoms-render
        ├── cpu    reference implementation (no dependencies)
        ├── image  RgbaImage, comparison, PNG
        ├── volume VolumeData, TransferFunction
        └── gpu    wgpu context, resources, pipelines (feature `gpu`)
```

**The dependency never points the other way.** `atoms-render` does not depend on
`atoms-core` or `atoms-orbitals`; with default features it has *no dependencies at
all*. The physics crates do not know the renderer exists. The viewer and any future
bridge crate depend on both, and that is where the coupling lives.

`atoms-render`'s own tests and examples do depend on the physics crates, as Cargo
`dev-dependencies`. That is deliberate and does not alter the library's dependency
graph: dev-dependencies are not propagated to consumers.

## 3. The interface: `VolumeData`

```rust
pub struct VolumeData {
    shape: [usize; 3],          // (nx, ny, nz)
    origin_bohr: [f64; 3],      // coordinate of grid point (0,0,0)
    spacing_bohr: [f64; 3],     // step per axis
    values: Vec<f32>,           // public array order
    value_unit: String,         // e.g. "bohr^-3"
}
```

### Axis order (normative)

`values` is in the **public array order** used everywhere in this workspace: C order
over `(i, j, k)` with **`k` fastest**,

```
index(i, j, k) = (i * ny + j) * nz + k
```

so `values[index(i, j, k)]` is the field at `origin + (i, j, k) * spacing`. This is
the same order as the NumPy arrays the Python API returns, so **no transposition is
required anywhere in the data path**. `atoms_core::spatial_grid::SpatialGrid3D::index_public`
is the single definition of this layout; `atoms-python`'s ordering module and
`atoms-render` both defer to it.

### Units

Geometry is in **bohr**. Values carry the unit of the producing quantity — for a
probability density, `bohr^-3`, recorded in `value_unit` and written into the
artefact metadata. The renderer never converts units and never rescales values; it
only applies a transfer function for display.

### From a physics grid

```rust
fn to_volume(g: &SpatialGrid3D, values: &[f64]) -> Result<VolumeData, RenderError> {
    let o = g.origin();
    let s = g.spacing();
    VolumeData::from_f64(g.counts(), [o.x, o.y, o.z], [s.x, s.y, s.z],
                         values, "bohr^-3", true)
}
```

`values` comes from `g.map_public(|x, y, z| wf.probability_density(x, y, z))`.
`require_non_negative = true` makes a negative density a hard error rather than a
silently clamped pixel.

### f64 → f32

The GPU consumes `f32`. Two metrics are reported rather than assumed:

| metric | meaning | measured on the 2p_z scene |
|---|---|---|
| `max_relative_conversion_error` | worst per-value relative error | 1.166e-6 |
| `max_error_relative_to_field_max` | worst absolute error as a fraction of the field maximum | **4.299e-8** |

The first is large and that is expected: an orbital density spans ~40 orders of
magnitude, so the nodal-plane residue (~1e-35) is *subnormal* in `f32` and its
relative error is correspondingly poor. The second is the metric that bounds what a
renderer can see, because the transfer function normalises by the field maximum, and
it stays within `2^-24`. Both are printed by the example and asserted by tests.

## 4. Image orientation (normative)

For a slice or projection along axis `a`, the image uses the two remaining axes in
ascending order, with the later axis vertical and **increasing upwards** (row 0 is
the largest coordinate):

| `axis` | width | height | pixel `(px, py)` shows |
|---|---|---|---|
| `X` | `ny` | `nz` | `(i = fixed, j = px, k = nz-1-py)` |
| `Y` | `nx` | `nz` | `(i = px, j = fixed, k = nz-1-py)` |
| `Z` | `nx` | `ny` | `(i = px, j = ny-1-py, k = fixed)` |

The WGSL shader and `atoms_render::cpu` implement the same rule; tests assert they
agree pixel for pixel on an asymmetric volume (11×7×5, distinct values per axis),
which is the only shape that can detect a transposition or an axis swap.

## 5. Channels

Rendered images are `RGBA8`:

* **RGB** — display colour (viridis, five anchors, linear interpolation).
* **A** — the scalar intensity itself, quantised to `0..=255`, **without** any
  colour map.

Validation compares the **alpha channel only**. A colour-map difference therefore
cannot be mistaken for a physics difference, and the interesting number is a direct
function of the field.

The render target is `Rgba8Unorm`, not `-Srgb`, so no automatic transfer-function
encoding is applied: the bytes in memory are exactly the values the shader wrote.

## 6. Transfer function (normative)

```
t     = clamp(value / vmax, 0, 1)
n     = ln(1 + t * K) / ln(1 + K)      K = 20
byte  = floor(n * 255 + 0.5)
```

`vmax` is fixed by the caller (normally the volume maximum) so that two slices of
the same volume share a dynamic range and remain comparable. `K = 0` degenerates to
a linear ramp; the formula would otherwise evaluate `ln(1)/ln(1) = 0/0`, which is
why that case is handled explicitly on both sides.

Rounding uses `floor(x + 0.5)` rather than a `round` intrinsic because `round` has
implementation-defined tie behaviour in WGSL while `floor` does not.

## 7. Resource ownership

One rule: **the context owns the device and queue, and every resource owns what it
creates.**

| object | owner | lifetime |
|---|---|---|
| `wgpu::Instance`, `Adapter` | local to `select_adapter` | dropped once the device exists |
| `Device`, `Queue` | `GpuContext`, behind `Arc` | as long as any clone of the context |
| `BindGroupLayout`, `RenderPipeline` | `SliceRenderer` | as long as the renderer |
| volume storage buffer | caller of `upload_volume`, or the render call | as long as the caller keeps it |
| uniform buffer, render target, bind group | the render call | dropped when the call returns |
| staging buffer | the render call | dropped after the mapped data is copied out and the buffer unmapped |

`GpuContext` is `Clone` and `Send + Sync`; a clone keeps the device alive for as long
as the clone exists, so a resource can never outlive its device. `SliceRenderer`
holds no reference to the context, so it can be moved between threads. Nothing
borrows from a temporary.

`upload_volume` and `render_with_buffer` are separate from `render_slice`/`render_mip`
so that rendering many views of one volume pays the upload once, and so that a
benchmark can measure upload and render cost independently.

## 8. Failure handling

Nothing panics because a capability is missing. Every rejection is a typed
`RenderError`:

| situation | error | classified as |
|---|---|---|
| no adapter at all | `NoAdapter` | unavailable |
| only a software adapter, and it was disallowed | `NoAdapter` | unavailable |
| device creation failed | `DeviceCreationFailed` | unavailable |
| a required feature is missing | `UnsupportedCapability` | capability mismatch |
| a required limit exceeds the adapter's | `ResourceLimit` | capability mismatch |
| a volume exceeds the storage-binding limit | `ResourceLimit` | capability mismatch |
| an image exceeds `max_texture_dimension_2d` | `ResourceLimit` | capability mismatch |
| readback failed | `ReadbackFailed` | error |

`RenderError::is_unavailable()` lets a caller fall back to the CPU path without
treating "no GPU here" as a failure. `check_required_features` and
`check_required_limits` are pure functions over adapter capabilities, so the policy
is unit-tested on machines that have no GPU at all.

## 9. Threading

`request_context` is asynchronous and should be awaited from whatever scheduler the
host uses. `init_blocking` is a `pollster` wrapper documented for **worker threads**:
it blocks, which is the reason the async entry point exists. A UI thread must never
call it.

## 10. Backend status

Reported from actual execution, not from compilation:

| backend | platform | status | evidence |
|---|---|---|---|
| **Metal** | macOS 26.7.1, Apple M4 Max | **tested** | adapter `Apple M4 Max`, `integrated-gpu`; 68 GPU-path tests pass; 2.8 M-point volume uploaded and rendered |
| Vulkan | not available on this machine | **untested** | no Vulkan-capable device; `wgpu` selects Metal on macOS |
| MoltenVK | not installed | **untested** | not evaluated |
| DX12 | Windows | **untested** | no Windows machine available |
| GL/WebGL | not attempted | **untested** | not a target for offscreen rendering |
| native Vulkan (`ash`) | — | **not implemented** | deferred until a portable baseline and profiling justify it |
| CUDA | — | **not implemented** | deferred; `atoms-gpu` keeps the seam |

Compilation is not evidence of correctness, and no backend other than Metal has been
exercised. Vulkan and the other backends remain to be verified on suitable hardware.

### Device limits observed (Apple M4 Max, Metal)

```
max texture 2d/3d    8192 / 2048
max buffer            268435456 bytes (256 MiB)
max storage binding   134217728 bytes (128 MiB)
```

A 141³ `f32` volume is 11.2 MB, well inside both limits. The limits are checked
before allocation and an oversized volume returns `ResourceLimit` rather than
failing inside the driver.

## 11. Validation

The claim "the GPU agrees with the CPU" is supported by:

1. **Upload integrity** — the volume buffer is read back and compared against the
   CPU's array by ULP difference. Measured: **max ULP = 0**, i.e. bit-identical.
2. **Rendered-image agreement** — the alpha channel of the GPU image against the CPU
   image, on a real 2p_z density (2.8 M points) for slices on all three axes and MIPs
   on all three axes. Measured: **max difference 0 of 255, every pixel identical**.
   The declared tolerance is 1 quantisation step, and the reason it is not 0 is
   honest: the two sides evaluate `ln` in different implementations, and a value
   sitting exactly on a quantisation boundary could flip. On this data it does not.
3. **An asymmetric synthetic volume** (11×7×5, values encoding their own coordinates,
   every axis and every index) for transposition and axis-swap detection.
4. **Exact physics anchors** — the nodal plane of 2p_z must render as an entirely
   black image (asserted on both CPU and GPU), and the brightest pixel of the peak
   slice must be the point on the axis.
5. **Determinism** — rendering the same volume twice is asserted bit-stable.

Tolerances are declared in the tests themselves, together with the reason for the
value chosen.

## 12. Measured performance

Release build (`lto = true`, `codegen-units = 1`), Apple M4 Max, Metal, 141³ volume
(11.2 MB, 19881 px images), three runs:

| operation | measured |
|---|---|
| field evaluation, 2.8 M points (CPU) | 78–105 ms |
| f64 → `VolumeData` (2.8 M points) | 4.1–5.9 ms |
| volume upload to device (11.2 MB) | 1.8–2.1 ms |
| three images on the GPU, readback included | 21–28 ms |
| three images on the CPU reference | 6.2–8.7 ms |
| *(debug build, same work, for scale)* | 561 ms / 46 ms / 18 ms / 96 ms |

### The GPU path is slower than the CPU path for this workload

**21–28 ms on the GPU against 6–9 ms on one CPU thread.** The CPU is ~3–4× faster
for three images at this size, and that is the honest headline.

The reason is that each GPU image pays for a synchronised round trip — render-target
creation, submission, staging buffer, texture-to-buffer copy, device poll and map —
roughly 7 ms per image, while the CPU writes pixels into a `Vec` in ~2 ms.

This has three consequences for the architecture:

1. **Offscreen render + readback is a validation harness, not a performance path.**
   It exists to put the GPU's pixels next to a reference's, which requires them on
   the host. A viewer presents to a surface and never reads back; once readback is
   removed this comparison no longer applies.
2. **Per-call overhead dominates at this image size, not bandwidth or shading.**
   Amortising means keeping the volume resident (`upload_volume` +
   `render_with_buffer` already do) and reusing a persistent render target.
3. **The dominant cost is the field, not the rendering** — 78–105 ms to evaluate
   versus 6–9 ms for all rendering. That is where GPU compute would actually matter,
   and it is M5's question.

The GPU figures include render-target creation, submission, readback and mapping for
three images, so they are wall-clock figures for the whole round trip, not kernel
times. There is no GPU compute kernel, so nothing here says anything about GPU
acceleration of a calculation.

One further caveat on fairness: the comparison above is between a *synchronised*
GPU round trip and an in-process CPU loop. It is the right comparison for the
validation harness, because that is what the harness does. It is **not** the right
comparison for an interactive viewer, which presents to a surface and never reads
back — and therefore never pays the ~7 ms per image that dominates the GPU column.

### The wording used elsewhere in this repository

Because the measurement above is negative, the phrase "GPU-accelerated" is not used
for this milestone's rendering. The accurate statement is: *a portable GPU rendering
path exists, is validated against a CPU reference, and is not faster than the CPU
reference for the workloads measured so far.* No performance claim is made in either
direction beyond the numbers in this section.

## 13. Known limitations

* **Slices and axis-aligned projections only.** The scene is rendered from a fixed
  direction along a grid axis. There is no camera, no interpolation, and no
  view-dependent rendering. Trilinear sampling and arbitrary view directions belong
  to M2-C.
* **Nearest-neighbour sampling.** A slice plane coincides with voxel centres, so
  nearest-neighbour is exact for this operation; it would not be for an arbitrary
  view.
* **Volume rendering** (ray marching, isosurfaces, transfer-function gradients) is
  not implemented.
* **3-D textures are unused.** The volume travels as a storage buffer. A `texture_3d`
  path is the natural evolution once interpolation is needed; it was not chosen now
  because a storage buffer has one owner, is trivially read back for verification,
  and avoids filtering-capability questions entirely.
* **No timestamp queries or profiling counters.** The timings above are wall-clock.
* **Single render target, no depth, no MSAA.**
* **The PNG encoder does not compress.** Files are ~5× larger than a compressing
  encoder's. It exists so artefacts can be produced without an image dependency.
* **Softmax/parallelism inside the CPU reference** is absent: it is single-threaded
  by design, so its cost is reported as the reference cost, not as the best CPU cost.
* **The GPU path is not faster than the CPU path at the sizes measured.** See §12.
  The offscreen-plus-readback design is a validation harness; an interactive viewer
  would not pay the readback cost, but that has not been measured and is not claimed.
