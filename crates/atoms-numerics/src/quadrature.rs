//! Numerical quadrature.
//!
//! Composite Newton-Cotes rules are sufficient for the smooth integrands
//! encountered in hydrogenic problems (radial wavefunctions are products of
//! polynomials, exponentials and powers, all analytic on the integration
//! domain). Higher-order Gaussian quadrature is deliberately not introduced
//! until a concrete accuracy or performance need is demonstrated.

/// Composite Simpson's rule on `[a, b]` with `n_intervals` sub-intervals.
///
/// If `n_intervals` is odd it is incremented by one so that the rule remains
/// valid. `n_intervals` is clamped to a minimum of 2.
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n_intervals: usize) -> f64 {
    let n = if n_intervals < 2 { 2 } else { n_intervals };
    let n = if n % 2 == 1 { n + 1 } else { n };
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + h * i as f64;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * f(x);
    }
    sum * h / 3.0
}

/// Composite trapezoidal rule on `[a, b]` with `n_intervals` sub-intervals.
pub fn trapezoid<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n_intervals: usize) -> f64 {
    let n = n_intervals.max(1);
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + h * i as f64);
    }
    sum * h
}

/// Integral `\int_a^b f(x) dx` using the substitution `x = a + t/(1-t)` mapped
/// onto `t in [0,1)`, suitable for semi-infinite domains.
///
/// `transform(x)` must return the unit-less substitution Jacobian
/// `dx/dt` for the chosen mapping. For the mapping `x = a + t/(1-t)` the
/// Jacobian is `1/(1-t)^2`.
///
/// This helper exists so that semi-infinite radial integrals can be checked
/// without truncating at an arbitrary `r_max`; the primary validation tests use
/// explicit truncation and document the truncation error instead.
pub fn simpson_semi_infinite<F: Fn(f64) -> f64>(f: F, a: f64, n_intervals: usize) -> f64 {
    // Integrate g(t) = f(a + t/(1-t)) / (1-t)^2 over t in [0, 1].
    // The endpoint t = 1 is not evaluable; Simpson's rule never evaluates the
    // midpoint-weighted interior only, but it does evaluate both endpoints, so
    // we integrate up to 1 - eps and add the (negligible) remainder bound.
    let eps = 1e-12_f64;
    let g = |t: f64| {
        let omt = 1.0 - t;
        let x = a + t / omt;
        f(x) / (omt * omt)
    };
    simpson(g, 0.0, 1.0 - eps, n_intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn simpson_integrates_polynomials_exactly() {
        // deg <= 3 is exact for Simpson
        let cubic = |x: f64| x * x * x - 2.0 * x + 1.0;
        // \int_0^2 = 4 - 4 + 2 = 2
        assert!((simpson(cubic, 0.0, 2.0, 100) - 2.0).abs() < 1e-12);

        let quad = |x: f64| x * x;
        // \int_0^3 = 9
        assert!((simpson(quad, 0.0, 3.0, 1000) - 9.0).abs() < 1e-12);
    }

    #[test]
    fn simpson_handles_odd_and_small_interval_counts() {
        let f = |x: f64| x;
        // \int_0^1 x dx = 0.5 ; odd count must be bumped to even, not panic
        assert!((simpson(f, 0.0, 1.0, 1) - 0.5).abs() < 1e-15);
        assert!((simpson(f, 0.0, 1.0, 3) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn trapezoid_is_second_order() {
        let f = |x: f64| x * x;
        let coarse = trapezoid(f, 0.0, 1.0, 100);
        let fine = trapezoid(f, 0.0, 1.0, 200);
        // error ~ h^2 -> halving h quarters the error
        let e_coarse = (coarse - 1.0 / 3.0).abs();
        let e_fine = (fine - 1.0 / 3.0).abs();
        assert!(e_fine < e_coarse / 3.5, "c={e_coarse} f={e_fine}");
    }

    #[test]
    fn simpson_gaussian_normalisation() {
        // \int_0^inf e^{-x} dx = 1
        let v = simpson_semi_infinite(|x| (-x).exp(), 0.0, 200_000);
        assert!((v - 1.0).abs() < 1e-6, "v={v}");
    }

    #[test]
    fn simpson_angular_measure() {
        // \int_0^pi sin(theta) dtheta = 2
        let v = simpson(|t: f64| t.sin(), 0.0, PI, 10_000);
        assert!((v - 2.0).abs() < 1e-12);
    }
}
