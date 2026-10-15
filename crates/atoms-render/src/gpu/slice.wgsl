// Slice and maximum-intensity-projection renderer for a scalar volume.
//
// The volume arrives as a flat storage buffer in the workspace's public array
// order: index(i, j, k) = (i * ny + j) * nz + k, with k fastest. See
// `atoms_render::volume::VolumeData`.
//
// Output conventions (must match `atoms_render::cpu`, which is the reference):
//   * the pixel at fragment position (px, py) shows the grid point
//     (fixed = index, h = px, v = height - 1 - py) where the two image axes are
//     the remaining axes of the volume in ascending order;
//   * the alpha channel carries the quantised scalar intensity, 0..=255, with no
//     colour map applied;
//   * the RGB channels carry the viridis display colour of that intensity.
//
// The transfer function here is a transcription of
// `atoms_render::volume::TransferFunction`. Both use floor(x + 0.5) for
// rounding because WGSL `round` has implementation-defined tie behaviour.

struct Uniforms {
    // nx, ny, nz, unused
    shape: vec4<u32>,
    // mode (0 = slice, 1 = mip), axis (0 = x, 1 = y, 2 = z), index, depth
    fixed: vec4<u32>,
    // vmax, compression, unused, unused
    params: vec4<f32>,
    // width, height, unused, unused
    out_size: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> volume: array<f32>;

// Viridis anchors, identical to `atoms_render::image::VIRIDIS_ANCHORS`.
// Written as a branch chain rather than an indexed array so that no dynamic
// indexing of a constant array is required by the backend.
fn viridis_anchor(idx: u32) -> vec3<f32> {
    if (idx == 0u) {
        return vec3<f32>(0.267004, 0.004874, 0.329415);
    }
    if (idx == 1u) {
        return vec3<f32>(0.229739, 0.322361, 0.545706);
    }
    if (idx == 2u) {
        return vec3<f32>(0.127568, 0.566949, 0.550556);
    }
    if (idx == 3u) {
        return vec3<f32>(0.369214, 0.788888, 0.382914);
    }
    return vec3<f32>(0.993248, 0.906157, 0.143936);
}

// `t` is taken from the already-quantised intensity so that the colour is a
// function of the same integer the CPU reference uses.
fn colour_map(scalar: u32) -> vec3<f32> {
    let t = f32(scalar) / 255.0;
    let bands = 4.0;
    let pos = t * bands;
    let lo_f = floor(pos);
    let lo = u32(lo_f);
    let frac = pos - lo_f;
    let c0 = viridis_anchor(lo);
    let c1 = viridis_anchor(min(lo + 1u, 4u));
    // Clamped so a shader can never write outside [0, 1].
    return clamp(c0 * (1.0 - frac) + c1 * frac, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_index(i: u32, j: u32, k: u32) -> u32 {
    let ny = u.shape.y;
    let nz = u.shape.z;
    return (i * ny + j) * nz + k;
}

fn grid_index(axis: u32, fixed_coord: u32, px: u32, py: u32) -> vec3<u32> {
    let v = u.out_size.y - 1u - py;
    if (axis == 0u) {
        return vec3<u32>(fixed_coord, px, v);
    }
    if (axis == 1u) {
        return vec3<u32>(px, fixed_coord, v);
    }
    return vec3<u32>(px, v, fixed_coord);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // One oversized triangle covering the whole viewport; no vertex buffer.
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(corners[vertex_index], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let px = u32(frag.x);
    let py = u32(frag.y);
    let axis = u.fixed.y;
    let index = u.fixed.z;
    let depth = u.fixed.w;

    var value: f32 = 0.0;
    if (u.fixed.x == 0u) {
        let g = grid_index(axis, index, px, py);
        value = volume[linear_index(g.x, g.y, g.z)];
    } else {
        var best: f32 = -1.0e30;
        for (var d: u32 = 0u; d < depth; d = d + 1u) {
            let g = grid_index(axis, d, px, py);
            best = max(best, volume[linear_index(g.x, g.y, g.z)]);
        }
        value = max(best, 0.0);
    }

    var t: f32 = 0.0;
    if (u.params.x > 0.0) {
        t = clamp(value / u.params.x, 0.0, 1.0);
    }
    var n: f32 = t;
    if (u.params.y > 0.0) {
        n = log(1.0 + t * u.params.y) / log(1.0 + u.params.y);
    }
    let scalar = u32(clamp(floor(n * 255.0 + 0.5), 0.0, 255.0));

    return vec4<f32>(colour_map(scalar), f32(scalar) / 255.0);
}
