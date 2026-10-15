//! Analytically known physical results.
//!
//! Every assertion in this file is anchored to a closed-form result of the
//! Coulomb problem, not to a previously recorded output of the code under test.
//! Where a numerical approximation (grid integration, quadrature) is involved,
//! the measured deviation is printed and the tolerance is justified.

use atoms_core::coordinates::spherical_to_cartesian;
use atoms_core::spatial_grid::SpatialGrid3D;
use atoms_core::EnergyLevel;
use atoms_numerics::quadrature::simpson;
use atoms_orbitals::expectations as exact;
use atoms_orbitals::{
    radial_expectation_value, DensityGrid, HydrogenWavefunction, QuantumNumbers, SphericalHarmonic,
};
use nalgebra::{Point3, Vector3};
use std::f64::consts::PI;

/// Grid used for volumetric checks: +/-12 bohr at 0.1875 bohr spacing.
///
/// The truncation error is `P(r > 12) < 1e-9` for every state tested, so the
/// only error source is the Riemann sum, which is `O(dx^2)`.
fn volumetric_grid() -> SpatialGrid3D {
    SpatialGrid3D::new(
        Point3::new(-12.0, -12.0, -12.0),
        Vector3::new(0.1875, 0.1875, 0.1875),
        [129, 129, 129],
    )
    .unwrap()
}

#[test]
fn energy_eigenvalues_follow_the_closed_form() {
    for n in 1..=20u32 {
        let e = EnergyLevel::new(n, 1).unwrap();
        let expected = -0.5 / (n as f64 * n as f64);
        assert!(
            (e.hartree - expected).abs() < 1e-16,
            "n={n}: {} vs {expected}",
            e.hartree
        );
        // Independent statement of the same physics: E = -<1/r>/2 (virial).
        let virial = -exact::expectation_inverse_r(n) / 2.0;
        assert!((e.hartree - virial).abs() < 1e-16, "n={n}");
    }
}

/// Energy depends only on `n`: the `l`-degeneracy of the pure Coulomb problem.
#[test]
fn energy_is_degenerate_in_l() {
    for n in 2..=8u32 {
        let reference = EnergyLevel::new(n, 1).unwrap().hartree;
        for l in 0..n {
            for m in -(l as i32)..=(l as i32) {
                let wf = HydrogenWavefunction::new(n, l, m).unwrap();
                assert_eq!(wf.energy().hartree, reference, "n={n} l={l} m={m}");
            }
        }
        // number of states in the shell is n^2
        assert_eq!(QuantumNumbers::shell(n).len(), (n * n) as usize);
    }
}

/// `int |psi|^2 dV = 1` checked by 3-D Riemann integration.
///
/// # Why only compact states are checked this way
///
/// A 3-D Cartesian Riemann sum needs an extent of order `<r> ~ n^2 / Z` bohr and
/// a spacing fine enough to resolve the radial nodes, so the number of points
/// grows roughly as `n^9`. For `n = 4` that is already more than `10^9` points.
/// Volumetric normalisation is therefore verified here for low-`n` states, and
/// the general statement `int r^2 R^2 dr * int |Y|^2 dOmega = 1` is verified for
/// all states by the marginal decomposition tests, which are cheap and exact.
#[test]
fn wavefunctions_are_normalised_over_a_grid() {
    let grid = volumetric_grid();
    const TOL: f64 = 5e-3;
    let states = [(1u32, 0u32, 0i32), (2, 0, 0), (2, 1, 0), (2, 1, 1)];
    let mut worst = 0.0_f64;
    for (n, l, m) in states {
        let wf = HydrogenWavefunction::new(n, l, m).unwrap();
        let d = DensityGrid::probability_density(&wf, &grid);
        let total = d.total_probability();
        worst = worst.max((total - 1.0).abs());
        assert!(
            (total - 1.0).abs() < TOL,
            "n={n} l={l} m={m}: integral = {total}"
        );
    }
    println!("max |int|psi|^2 dV - 1| on the volumetric grid: {worst:.3e}");
}

#[test]
fn probability_density_is_non_negative_and_finite_everywhere() {
    let grid = volumetric_grid();
    for n in 1..=4u32 {
        for l in 0..n {
            let m = (l as i32).min(1);
            let wf = HydrogenWavefunction::new(n, l, m).unwrap();
            let d = DensityGrid::probability_density(&wf, &grid);
            for &v in d.values() {
                assert!(v.is_finite(), "n={n} l={l} m={m}");
                assert!(v >= 0.0, "n={n} l={l} m={m}: {v}");
            }
        }
    }
}

/// Radial nodal structure `n - l - 1` and angular nodal structure `l`.
#[test]
fn nodal_structure_matches_the_quantum_numbers() {
    // Radial: count sign changes of R_{nl} over a fine radial scan.
    for n in 1..=6u32 {
        for l in 0..n {
            let expected = n - l - 1;
            let r_max = 4.0 * (n * n) as f64 + 20.0;
            let steps = 200_000;
            let dr = r_max / steps as f64;
            let mut changes = 0;
            let mut prev = 0i32;
            for i in 1..steps {
                let v = atoms_orbitals::radial_wavefunction(n, l, i as f64 * dr).unwrap();
                let s = if v > 0.0 {
                    1
                } else if v < 0.0 {
                    -1
                } else {
                    0
                };
                if s != 0 {
                    if prev != 0 && s != prev {
                        changes += 1;
                    }
                    prev = s;
                }
            }
            assert_eq!(changes, expected, "radial nodes n={n} l={l}");
        }
    }

    // Angular: the zonal harmonic Y_l^0 has exactly l sign changes in (0, pi).
    for l in 0..=6u32 {
        let steps = 20_000;
        let mut changes = 0;
        let mut prev = 0i32;
        for i in 1..steps {
            let theta = PI * i as f64 / steps as f64;
            let v = atoms_orbitals::spherical_harmonics::evaluate(l, 0, theta, 0.0).re;
            let s = if v > 0.0 {
                1
            } else if v < 0.0 {
                -1
            } else {
                0
            };
            if s != 0 {
                if prev != 0 && s != prev {
                    changes += 1;
                }
                prev = s;
            }
        }
        assert_eq!(changes, l, "angular nodes l={l}");
    }
}

/// The `m = 0` p orbital has the `z = 0` plane as a node; `|m| = 1` p orbitals
/// vanish on the `z` axis. Neither may be confused with the other.
#[test]
fn m_zero_and_m_one_p_orbitals_have_different_nodal_sets() {
    let grid = volumetric_grid();
    let two_p_z = HydrogenWavefunction::new(2, 1, 0).unwrap();
    let two_p_pm = HydrogenWavefunction::new(2, 1, 1).unwrap();

    // z = 0 plane: find the layer closest to z = 0
    let [nx, ny, nz] = grid.counts();
    let k_mid = nz / 2;
    let mid_z = grid.point(nx / 2, ny / 2, k_mid).z;
    assert!(mid_z.abs() < 1e-12, "mid-grid plane is not z = 0: {mid_z}");

    let d_z = DensityGrid::probability_density(&two_p_z, &grid);
    let d_pm = DensityGrid::probability_density(&two_p_pm, &grid);
    let slice_z = d_z.cross_section_xy(k_mid);
    let slice_pm = d_pm.cross_section_xy(k_mid);
    assert!(
        slice_z.iter().all(|&v| v < 1e-30),
        "2p_z must vanish in the z = 0 plane"
    );
    assert!(
        slice_pm.iter().any(|&v| v > 1e-6),
        "2p with |m| = 1 must not vanish in the z = 0 plane"
    );

    // z axis: 2p_{m=1} vanishes, 2p_z does not
    let axis_x = nx / 2;
    let axis_y = ny / 2;
    let max_pm = (0..nz)
        .map(|k| d_pm.value_at(axis_x, axis_y, k))
        .fold(0.0, f64::max);
    let max_z = (0..nz)
        .map(|k| d_z.value_at(axis_x, axis_y, k))
        .fold(0.0, f64::max);
    assert!(max_pm < 1e-30, "2p_{{m=1}} must vanish on the z axis");
    assert!(max_z > 1e-3, "2p_z must not vanish on the z axis");
}

/// Expectation values against the Kramers-Pasternack closed forms.
#[test]
fn expectation_values_match_closed_forms() {
    let cases: [(u32, u32); 8] = [
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 2),
        (4, 1),
        (5, 4),
        (6, 2),
    ];
    let mut worst_r = 0.0_f64;
    let mut worst_r2 = 0.0_f64;
    for (n, l) in cases {
        let wf = HydrogenWavefunction::new(n, l, 0).unwrap();

        let r = radial_expectation_value(&wf, 1);
        let r_ref = exact::expectation_r(n, l);
        worst_r = worst_r.max((r - r_ref).abs() / r_ref);
        assert!(
            (r - r_ref).abs() < 1e-6 * r_ref,
            "<r> n={n} l={l}: numeric={r} analytic={r_ref}"
        );

        let r2 = radial_expectation_value(&wf, 2);
        let r2_ref = exact::expectation_r_squared(n, l);
        worst_r2 = worst_r2.max((r2 - r2_ref).abs() / r2_ref);
        assert!(
            (r2 - r2_ref).abs() < 1e-5 * r2_ref,
            "<r^2> n={n} l={l}: numeric={r2} analytic={r2_ref}"
        );

        let inv_r = radial_expectation_value(&wf, -1);
        assert!(
            (inv_r - exact::expectation_inverse_r(n)).abs() < 1e-7,
            "<1/r> n={n} l={l}: {inv_r}"
        );

        let inv_r2 = radial_expectation_value(&wf, -2);
        let inv_r2_ref = exact::expectation_inverse_r_squared(n, l);
        assert!(
            (inv_r2 - inv_r2_ref).abs() < 1e-5 * inv_r2_ref,
            "<1/r^2> n={n} l={l}: numeric={inv_r2} analytic={inv_r2_ref}"
        );
    }
    println!("max relative <r> deviation: {worst_r:.3e}, <r^2>: {worst_r2:.3e}");
}

/// Coordinate consistency between the spherical and Cartesian entry points.
///
/// # Tolerance
///
/// The round trip `(r, theta, phi) -> (x, y, z) -> (r, theta, phi)` is exact to
/// about one machine epsilon, but the *density* amplifies an angular error by
/// the logarithmic derivative `d ln|Y|^2 / d theta`, which reaches ~17 for the
/// `l = 3` states sampled here. The achievable relative agreement is therefore
/// ~`17 * eps ~ 4e-15`; `1e-12` is used to leave a margin while remaining far
/// tighter than any real convention error, which would be of order 1.
#[test]
fn cartesian_and_spherical_evaluation_agree() {
    for n in 1..=4u32 {
        for l in 0..n {
            for m in -(l as i32)..=(l as i32) {
                let wf = HydrogenWavefunction::new(n, l, m).unwrap();
                for &(r, t, p) in &[
                    (0.3_f64, 0.2_f64, 0.1_f64),
                    (2.0, 1.3, 4.1),
                    (7.5, 2.8, 5.7),
                ] {
                    let c = spherical_to_cartesian(r, t, p);
                    let a = wf.value_spherical(r, t, p);
                    let b = wf.value_cartesian(c.x, c.y, c.z);
                    assert!(
                        (a - b).norm() < 1e-14,
                        "n={n} l={l} m={m} r={r}: {a} vs {b}"
                    );
                    let rho_a = wf.probability_density_spherical(r, t, p);
                    let rho_b = wf.probability_density(c.x, c.y, c.z);
                    assert!(
                        (rho_a - rho_b).abs() < 1e-12 * rho_a.max(1e-12),
                        "n={n} l={l} m={m} r={r}: rho {rho_a} vs {rho_b}"
                    );
                }
            }
        }
    }
}

/// `|psi|^2 = |R_{nl}(r)|^2 * |Y_l^m(theta)|^2`, i.e. radial and angular parts
/// factorise exactly.
#[test]
fn radial_and_angular_parts_factorise() {
    for (n, l, m) in [(2u32, 1u32, -1i32), (3, 2, 2), (4, 3, -3), (5, 1, 0)] {
        let wf = HydrogenWavefunction::new(n, l, m).unwrap();
        for &(r, t, p) in &[
            (0.4_f64, 0.3_f64, 1.0_f64),
            (3.0, 1.9, 2.7),
            (14.0, 0.9, 5.5),
        ] {
            let rad = atoms_orbitals::radial_wavefunction(n, l, r).unwrap();
            let expected = rad * rad * wf.angular_probability_density(t);
            let got = wf.probability_density_spherical(r, t, p);
            assert!(
                (got - expected).abs() <= 1e-15 * expected.max(1e-30),
                "n={n} l={l} m={m}: {got} vs {expected}"
            );
        }
    }
}

/// Hydrogenic `Z` scaling: energies scale as `Z^2`, radii as `1/Z`, and the
/// normalisation is preserved.
#[test]
fn hydrogenic_z_scaling_is_consistent() {
    for z in 1..=4u32 {
        // Energy
        let e = EnergyLevel::new(1, z).unwrap().hartree;
        assert!((e + 0.5 * (z as f64).powi(2)).abs() < 1e-15, "z={z}");
        // <r> scales as 1/Z for the 1s state
        let wf = HydrogenWavefunction::hydrogenic(1, 0, 0, z).unwrap();
        let r = radial_expectation_value(&wf, 1);
        assert!(
            (r - 1.5 / z as f64).abs() < 1e-6,
            "z={z}: <r> = {r}, expected {}",
            1.5 / z as f64
        );
        // Normalisation preserved
        let total = simpson(
            |x| wf.radial_probability_density(x),
            0.0,
            8.0 / z as f64 + 20.0,
            200_000,
        );
        assert!((total - 1.0).abs() < 1e-8, "z={z}: {total}");
    }
}

/// `sum_m |Y_l^m|^2 = (2l+1)/(4 pi)` exactly; a strong check of the `m < 0`
/// phase convention.
#[test]
fn spherical_harmonic_addition_theorem() {
    for l in 0..=8u32 {
        let expected = (2.0 * l as f64 + 1.0) / (4.0 * PI);
        for &theta in &[0.0, 0.31, 1.2, 2.4, PI] {
            let sum: f64 = (-(l as i32)..=l as i32)
                .map(|m| SphericalHarmonic::new(l, m).unwrap().abs_squared(theta))
                .sum();
            assert!(
                (sum - expected).abs() < 1e-12 * expected.max(1.0),
                "l={l} theta={theta}: {sum} vs {expected}"
            );
        }
    }
}

/// Angular orthonormality of the complex harmonics, integrated numerically over
/// the full solid angle.
#[test]
fn spherical_harmonics_are_orthonormal_over_the_sphere() {
    let states: Vec<(u32, i32)> = (0..=4u32)
        .flat_map(|l| {
            let lm = l as i32;
            (-lm..=lm).map(move |m| (l, m))
        })
        .collect();
    let nt = 240;
    let np = 240;
    for &(l1, m1) in &states {
        for &(l2, m2) in &states {
            let integral = simpson(
                |t| {
                    simpson(
                        |p| {
                            let a = atoms_orbitals::spherical_harmonics::evaluate(l1, m1, t, p);
                            let b = atoms_orbitals::spherical_harmonics::evaluate(l2, m2, t, p);
                            (a.conj() * b).re
                        },
                        0.0,
                        2.0 * PI,
                        np,
                    ) * t.sin()
                },
                0.0,
                PI,
                nt,
            );
            let expected = if (l1, m1) == (l2, m2) { 1.0 } else { 0.0 };
            assert!(
                (integral - expected).abs() < 1e-6,
                "l1={l1} m1={m1} l2={l2} m2={m2}: {integral}"
            );
        }
    }
}
