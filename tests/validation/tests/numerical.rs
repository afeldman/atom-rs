//! Numerical-method validation: convergence orders, sampler correctness,
//! reproducibility, and stability at the edges of the supported range.
//!
//! Tolerances here are tighter than in `physics.rs` because the quantities are
//! pure numerical-method properties with known asymptotic behaviour, not
//! discretisation-limited physical integrals.

use atoms_numerics::quadrature::{simpson, trapezoid};
use atoms_numerics::sampling::{rejection_sample_1d, InverseCdf1D, SeededRng};
use atoms_numerics::special::{associated_legendre_cs, generalized_laguerre};
use atoms_orbitals::{radial_wavefunction, HydrogenWavefunction};
use std::f64::consts::PI;

#[test]
fn simpson_is_exact_for_cubic_polynomials() {
    // Simpson's rule integrates polynomials of degree <= 3 exactly.
    for deg in 0..=3u32 {
        // int_0^2 x^deg dx = 2^(deg+1)/(deg+1)
        let expected = 2.0_f64.powi(deg as i32 + 1) / (deg as f64 + 1.0);
        let got = simpson(|x: f64| x.powi(deg as i32), 0.0, 2.0, 4);
        assert!(
            (got - expected).abs() < 1e-14,
            "deg={deg}: {got} vs {expected}"
        );
    }
}

#[test]
fn simpson_converges_at_fourth_order() {
    // int_0^1 e^x dx = e - 1 ; error ~ h^4 so doubling n cuts the error by 16.
    let exact = std::f64::consts::E - 1.0;
    let e1 = (simpson(|x: f64| x.exp(), 0.0, 1.0, 32) - exact).abs();
    let e2 = (simpson(|x: f64| x.exp(), 0.0, 1.0, 64) - exact).abs();
    let ratio = e1 / e2;
    assert!(
        (ratio - 16.0).abs() < 2.0,
        "error ratio = {ratio} (expected ~16), e1={e1:.3e} e2={e2:.3e}"
    );
    println!("Simpson error ratio on 32 -> 64 intervals: {ratio:.2} (expected 16)");
}

#[test]
fn trapezoid_converges_at_second_order() {
    let exact = std::f64::consts::E - 1.0;
    let e1 = (trapezoid(|x: f64| x.exp(), 0.0, 1.0, 512) - exact).abs();
    let e2 = (trapezoid(|x: f64| x.exp(), 0.0, 1.0, 1024) - exact).abs();
    let ratio = e1 / e2;
    assert!(
        (ratio - 4.0).abs() < 0.5,
        "error ratio = {ratio} (expected ~4)"
    );
}

#[test]
fn simpson_resolves_oscillatory_integrands() {
    // int_0^{2pi} sin^2(x) dx = pi
    let got = simpson(|x: f64| x.sin().powi(2), 0.0, 2.0 * PI, 200);
    assert!((got - PI).abs() < 1e-10, "got={got}");
}

/// Truncating the radial integral at `r_max` must incur a negligible error.
///
/// `r_max` is derived from the **analytic** `<r> = (3n^2 - l(l+1))/2` rather
/// than from a fixed multiple of `n^2`, because the radial extent of a state
/// depends on `l` as well as `n`. The measured deviation and the tail
/// probability beyond `r_max` are printed, so the truncation budget is visible
/// rather than assumed.
#[test]
fn radial_truncation_error_is_negligible() {
    for (n, l) in [(1u32, 0u32), (3, 2), (5, 0), (6, 5), (8, 0)] {
        let wf = HydrogenWavefunction::new(n, l, 0).unwrap();
        let mean_r = atoms_orbitals::expectations::expectation_r(n, l);
        let r_max = 8.0 * mean_r + 30.0;
        let total = simpson(|r| wf.radial_probability_density(r), 0.0, r_max, 400_000);

        // Tail mass beyond r_max, for the report.
        let tail = simpson(
            |r| wf.radial_probability_density(r),
            r_max,
            r_max + 8.0 * mean_r + 100.0,
            400_000,
        );
        println!(
            "n={n} l={l}: <r>={mean_r:.1}, r_max={r_max:.1} bohr, \
             integral={total:.15}, tail P(r>r_max)={tail:.3e}"
        );
        assert!(
            (total - 1.0).abs() < 1e-10,
            "n={n} l={l} r_max={r_max}: integral = {total}"
        );
    }
}

/// The inverse-CDF radial sampler must reproduce the analytic moments of `P(r)`.
#[test]
fn radial_sampler_reproduces_analytic_moments() {
    // 1s: P(r) = 4 r^2 e^{-2r}; <r> = 3/2, Var(r) = 3 - 9/4 = 3/4
    let pdf = |r: f64| {
        let v = radial_wavefunction(1, 0, r).unwrap();
        r * r * v * v
    };
    let sampler = InverseCdf1D::from_pdf(pdf, 0.0, 40.0, 40_000).expect("non-degenerate");
    let mut rng = SeededRng::from_seed(20261015);
    let n_samples = 300_000;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..n_samples {
        let r = sampler.sample(&mut rng);
        sum += r;
        sum_sq += r * r;
    }
    let mean = sum / n_samples as f64;
    let var = sum_sq / n_samples as f64 - mean * mean;
    // Standard error of the mean: sqrt(0.75 / 300000) ~ 0.0016
    assert!((mean - 1.5).abs() < 0.01, "mean = {mean}");
    assert!((var - 0.75).abs() < 0.02, "variance = {var}");
    println!("1s sampler: <r> = {mean:.5} (analytic 1.5), Var = {var:.5} (analytic 0.75)");

    // 2p: <r> = 5
    let pdf2 = |r: f64| {
        let v = radial_wavefunction(2, 1, r).unwrap();
        r * r * v * v
    };
    let sampler2 = InverseCdf1D::from_pdf(pdf2, 0.0, 60.0, 60_000).expect("non-degenerate");
    let mut sum = 0.0;
    for _ in 0..n_samples {
        sum += sampler2.sample(&mut rng);
    }
    let mean2 = sum / n_samples as f64;
    assert!((mean2 - 5.0).abs() < 0.03, "2p mean = {mean2}");
}

/// The angular sampler (cosine of the polar angle distributed as `|P_l^m|^2`)
/// must reproduce `<cos^2 theta>` computed by an entirely different method
/// (deterministic quadrature).
#[test]
fn angular_sampler_matches_deterministic_quadrature() {
    let mut rng = SeededRng::from_seed(4711);
    let n_samples = 400_000;
    for (l, m) in [(1u32, 0i32), (1, 1), (2, 0), (2, 2), (3, 1)] {
        let mm = m.unsigned_abs();
        let density = |mu: f64| {
            let p = associated_legendre_cs(l, mm, mu);
            p * p
        };
        // Quadrature reference on mu in [-1, 1]
        let norm = simpson(density, -1.0, 1.0, 200_000);
        let reference = simpson(|mu| mu * mu * density(mu), -1.0, 1.0, 200_000) / norm;

        let sampler = InverseCdf1D::from_pdf(density, -1.0, 1.0, 20_000).expect("valid");
        let mut acc = 0.0;
        for _ in 0..n_samples {
            let mu = sampler.sample(&mut rng);
            acc += mu * mu;
        }
        let sampled = acc / n_samples as f64;
        // standard error ~ sqrt(Var)/sqrt(N) < 1.5/632 ~ 0.0024, so 0.01 is > 4 sigma
        assert!(
            (sampled - reference).abs() < 0.01,
            "l={l} |m|={mm}: sampled={sampled} quadrature={reference}"
        );
        println!("l={l} |m|={mm}: <cos^2 theta> sampled={sampled:.5} quadrature={reference:.5}");
    }
}

#[test]
fn sampling_is_reproducible_for_a_fixed_seed() {
    let build = || {
        let pdf = |r: f64| {
            let v = radial_wavefunction(1, 0, r).unwrap();
            r * r * v * v
        };
        InverseCdf1D::from_pdf(pdf, 0.0, 40.0, 10_000).unwrap()
    };
    let a = build();
    let b = build();
    let mut ra = SeededRng::from_seed(99);
    let mut rb = SeededRng::from_seed(99);
    for _ in 0..5_000 {
        assert_eq!(a.sample(&mut ra), b.sample(&mut rb));
    }
    let mut rc = SeededRng::from_seed(100);
    let mut rd = SeededRng::from_seed(99);
    let first: Vec<f64> = (0..100).map(|_| a.sample(&mut rd)).collect();
    let second: Vec<f64> = (0..100).map(|_| a.sample(&mut rc)).collect();
    assert_ne!(first, second, "different seeds must differ");
}

#[test]
fn rejection_sampling_matches_the_tabulated_sampler() {
    let pdf = |x: f64| (-(x - 2.0).powi(2)).exp();
    let mut rng = SeededRng::from_seed(1234);
    let n = 100_000;
    let mut sum_rej = 0.0;
    for _ in 0..n {
        sum_rej += rejection_sample_1d(pdf, -2.0, 6.0, 1.0, &mut rng, 1000).unwrap();
    }
    let mean_rej = sum_rej / n as f64;
    let sampler = InverseCdf1D::from_pdf(pdf, -2.0, 6.0, 20_000).unwrap();
    let mut sum_cdf = 0.0;
    for _ in 0..n {
        sum_cdf += sampler.sample(&mut rng);
    }
    let mean_cdf = sum_cdf / n as f64;
    assert!(
        (mean_rej - mean_cdf).abs() < 0.02,
        "rejection={mean_rej} inverse-cdf={mean_cdf}"
    );
}

/// The Laguerre differential-equation check is a *definition* check: it verifies
/// that the recurrence computes the right polynomial family. It is therefore
/// restricted to low order, where a central finite difference is accurate.
///
/// At high order the finite-difference residual is dominated by truncation
/// error, because the derivatives of `L_k^a` grow roughly like `k!` while the
/// terms of the equation cancel down to zero. That is a property of the
/// *test method*, not of the implementation: accuracy at high order is
/// established in `reference_values.rs` against **exact rational arithmetic**,
/// which needs no differentiation at all.
#[test]
fn laguerre_satisfies_the_defining_differential_equation_at_low_order() {
    let h = 1e-3_f64;
    for k in 0..=6u32 {
        for &a in &[1.0_f64, 3.0, 5.0, 7.0, 9.0] {
            for &x in &[0.5, 2.0, 7.5, 20.0] {
                let l = generalized_laguerre(k, a, x);
                let lp = (generalized_laguerre(k, a, x + h) - generalized_laguerre(k, a, x - h))
                    / (2.0 * h);
                let lpp = (generalized_laguerre(k, a, x + h) - 2.0 * l
                    + generalized_laguerre(k, a, x - h))
                    / (h * h);
                let residual = x * lpp + (a + 1.0 - x) * lp + k as f64 * l;
                let scale = l.abs().max(1.0);
                assert!(
                    (residual / scale).abs() < 1e-5,
                    "k={k} a={a} x={x}: relative residual {}",
                    (residual / scale).abs()
                );
            }
        }
    }
}

/// The same recurrence evaluated at high order, in the region that actually
/// carries probability (`rho <= 4n`), must remain finite and finite-difference
/// differentiable. Quantitative accuracy at these orders is asserted against
/// exact rational arithmetic in `reference_values.rs`.
#[test]
fn laguerre_values_are_finite_and_smooth_at_high_order() {
    let h = 1e-3_f64;
    for (k, a) in [(30u32, 1.0_f64), (39, 1.0), (20, 41.0), (10, 61.0)] {
        for &x in &[0.5_f64, 5.0, 20.0, 40.0] {
            let v = generalized_laguerre(k, a, x);
            assert!(v.is_finite(), "k={k} a={a} x={x} -> {v}");
            // Smoothness: the second difference must stay bounded relative to
            // the local scale of the polynomial.
            let d2 = (generalized_laguerre(k, a, x + h) - 2.0 * v
                + generalized_laguerre(k, a, x - h))
                / (h * h);
            assert!(d2.is_finite(), "k={k} a={a} x={x} d2={d2}");
            assert!(
                d2.abs() < 1e18,
                "k={k} a={a} x={x}: second derivative {d2:e} is implausibly large"
            );
        }
    }
}

#[test]
fn legendre_stays_finite_at_high_degree() {
    for l in [10u32, 20, 30] {
        for m in [0, 1, l / 2, l] {
            for &x in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
                let v = associated_legendre_cs(l, m, x);
                assert!(v.is_finite(), "l={l} m={m} x={x} -> {v}");
            }
        }
    }
}

/// Simpson's rule on the angular measure must recover the exact value for the
/// uniform spherical distribution.
#[test]
fn angular_measure_is_normalised() {
    let total = simpson(|t: f64| t.sin(), 0.0, PI, 10_000);
    assert!((total - 2.0).abs() < 1e-12);
    // and the total solid angle is 4 pi
    assert!((total * 2.0 * PI - 4.0 * PI).abs() < 1e-11);
}
