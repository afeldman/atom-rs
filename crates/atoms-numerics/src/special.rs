//! Special functions required for hydrogenic wavefunctions.
//!
//! All functions operate in `f64` and are implemented with the standard
//! three-term recurrence relations. Recurrences are used deliberately: they are
//! numerically stable in the ranges required here (`k` up to ~40, `x` in the
//! classically allowed region) and avoid the catastrophic cancellation that
//! explicit closed-form polynomials exhibit for large `k`.

/// Exact factorial `n!` for `n <= 170` computed by iterative multiplication.
///
/// Returns `f64::INFINITY` above the representable range (`n > 170`).
///
/// # Panics
/// Never panics; overflow saturates to infinity.
pub fn factorial(n: u32) -> f64 {
    // 170! is the largest factorial that fits in f64.
    if n > 170 {
        return f64::INFINITY;
    }
    let mut acc = 1.0_f64;
    for i in 2..=n {
        acc *= i as f64;
    }
    acc
}

/// Double factorial `n!!` for `n >= 0`, with `(-1)!! = 1` and `0!! = 1`.
pub fn double_factorial(n: u32) -> f64 {
    let mut acc = 1.0_f64;
    let mut k = n;
    while k > 1 {
        acc *= k as f64;
        k -= 2;
    }
    acc
}

/// Generalized (associated) Laguerre polynomial `L_k^alpha(x)`.
///
/// Uses the standard recurrence
/// ```text
/// L_0^a(x)     = 1
/// L_1^a(x)     = 1 + a - x
/// j L_j^a(x)   = (2j - 1 + a - x) L_{j-1}^a(x) - (j - 1 + a) L_{j-2}^a(x)
/// ```
///
/// Note: this is the convention in which `L_k^alpha` is the polynomial used by
/// the hydrogen radial wavefunction, i.e. it is *not* multiplied by a binomial
/// prefactor and it is not normalised.
pub fn generalized_laguerre(k: u32, alpha: f64, x: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    let mut l_prev2 = 1.0_f64; // L_0^a
    let mut l_prev1 = 1.0 + alpha - x; // L_1^a
    if k == 1 {
        return l_prev1;
    }
    let mut l_cur = l_prev1;
    for j in 2..=k {
        let jf = j as f64;
        l_cur = ((2.0 * jf - 1.0 + alpha - x) * l_prev1 - (jf - 1.0 + alpha) * l_prev2) / jf;
        l_prev2 = l_prev1;
        l_prev1 = l_cur;
    }
    l_cur
}

/// Associated Legendre function `P_l^m(x)` **including** the Condon-Shortley
/// phase `(-1)^m`, for `0 <= m <= l` and `-1 <= x <= 1`.
///
/// Recurrence:
/// ```text
/// P_m^m(x)     = (-1)^m (2m-1)!! (1 - x^2)^{m/2}
/// P_{m+1}^m(x) = x (2m+1) P_m^m(x)
/// (l-m) P_l^m(x) = x (2l-1) P_{l-1}^m(x) - (l+m-1) P_{l-2}^m(x)
/// ```
///
/// The argument is clamped to `[-1, 1]` to absorb floating-point round-off from
/// `cos(theta)` evaluations.
pub fn associated_legendre_cs(l: u32, m: u32, x: f64) -> f64 {
    debug_assert!(
        m <= l,
        "associated Legendre requires m <= l (got l={l}, m={m})"
    );
    let x = x.clamp(-1.0, 1.0);
    let mf = m as f64;

    // P_m^m(x)
    let mut pmm = 1.0_f64;
    if m > 0 {
        let somx2 = (1.0 - x) * (1.0 + x);
        let root = somx2.max(0.0).sqrt();
        let mut fact = 1.0_f64;
        for _ in 1..=m {
            pmm *= -fact * root; // accumulate the (-1)^m Condon-Shortley phase
            fact += 2.0;
        }
    }
    if l == m {
        return pmm;
    }

    // P_{m+1}^m(x)
    let mut pmmp1 = x * (2.0 * mf + 1.0) * pmm;
    if l == m + 1 {
        return pmmp1;
    }

    // Upward recurrence in l.
    let mut pll = 0.0_f64;
    for ll in (m + 2)..=l {
        let llf = ll as f64;
        pll = (x * (2.0 * llf - 1.0) * pmmp1 - (llf + mf - 1.0) * pmm) / (llf - mf);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    pll
}

/// Associated Legendre function `P_l^m(x)` **without** the Condon-Shortley
/// phase, i.e. `(-1)^m * P_l^m_cs(x)`.
pub fn associated_legendre_no_cs(l: u32, m: u32, x: f64) -> f64 {
    let v = associated_legendre_cs(l, m, x);
    if m % 2 == 1 {
        -v
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factorial_matches_closed_form() {
        assert_eq!(factorial(0), 1.0);
        assert_eq!(factorial(1), 1.0);
        assert_eq!(factorial(5), 120.0);
        assert_eq!(factorial(10), 3628800.0);
        // 20! == 2_432_902_008_176_640_000 exactly representable
        assert!((factorial(20) - 2.43290200817664e18).abs() / 2.43290200817664e18 < 1e-15);
        assert!(factorial(200).is_infinite());
    }

    #[test]
    fn double_factorial_values() {
        assert_eq!(double_factorial(0), 1.0);
        assert_eq!(double_factorial(1), 1.0);
        assert_eq!(double_factorial(5), 15.0);
        assert_eq!(double_factorial(6), 48.0);
    }

    #[test]
    fn laguerre_low_order_closed_forms() {
        let x = 0.7_f64;
        assert_eq!(generalized_laguerre(0, 2.0, x), 1.0);
        // L_1^a(x) = 1 + a - x
        assert!((generalized_laguerre(1, 1.0, x) - (2.0 - x)).abs() < 1e-15);
        assert!((generalized_laguerre(1, 3.0, x) - (4.0 - x)).abs() < 1e-15);
        // L_2^a(x) = ((a+2)(a+1) - 2(a+2)x + x^2)/2
        let a = 1.0_f64;
        let expected = ((a + 2.0) * (a + 1.0) - 2.0 * (a + 2.0) * x + x * x) / 2.0;
        assert!((generalized_laguerre(2, a, x) - expected).abs() < 1e-14);
        // Explicit: L_2^3(x) = (x^2 - 10x + 20)/2
        let expected3 = (x * x - 10.0 * x + 20.0) / 2.0;
        assert!((generalized_laguerre(2, 3.0, x) - expected3).abs() < 1e-14);
    }

    #[test]
    fn laguerre_hydrogen_case_is_linear() {
        // L_{n-l-1}^{2l+1} for the 2s orbital: k=1, alpha=1 -> 2 - rho
        for rho in [0.0, 0.5, 2.0, 10.0] {
            assert!((generalized_laguerre(1, 1.0, rho) - (2.0 - rho)).abs() < 1e-14);
        }
    }

    #[test]
    fn legendre_closed_forms() {
        for &x in &[-1.0, -0.53, 0.0, 0.25, 1.0] {
            assert!((associated_legendre_cs(0, 0, x) - 1.0).abs() < 1e-15);
            assert!((associated_legendre_cs(1, 0, x) - x).abs() < 1e-15);
            // P_2^0 = (3x^2 - 1)/2
            assert!((associated_legendre_cs(2, 0, x) - (3.0 * x * x - 1.0) / 2.0).abs() < 1e-15);
            // P_3^0 = (5x^3 - 3x)/2
            assert!(
                (associated_legendre_cs(3, 0, x) - (5.0 * x * x * x - 3.0 * x) / 2.0).abs() < 1e-15
            );
            let s = (1.0 - x * x).max(0.0).sqrt();
            // P_1^1 = -sqrt(1-x^2)
            assert!((associated_legendre_cs(1, 1, x) + s).abs() < 1e-15);
            // P_2^1 = -3x sqrt(1-x^2)
            assert!((associated_legendre_cs(2, 1, x) + 3.0 * x * s).abs() < 1e-14);
            // P_2^2 = 3(1-x^2)
            assert!((associated_legendre_cs(2, 2, x) - 3.0 * (1.0 - x * x)).abs() < 1e-14);
            // P_3^3 = -15 (1-x^2)^{3/2}
            assert!((associated_legendre_cs(3, 3, x) + 15.0 * s * s * s).abs() < 1e-13);
        }
    }

    #[test]
    fn legendre_no_cs_phase_relation() {
        for &x in &[-0.8, 0.0, 0.42] {
            for l in 0..=4u32 {
                for m in 0..=l {
                    let a = associated_legendre_cs(l, m, x);
                    let b = associated_legendre_no_cs(l, m, x);
                    let expected = if m % 2 == 1 { -a } else { a };
                    assert!((b - expected).abs() < 1e-14, "l={l} m={m}");
                }
            }
        }
    }

    /// The associated Legendre functions satisfy the associated Legendre
    /// differential equation
    /// `(1-x^2) P'' - 2x P' + [l(l+1) - m^2/(1-x^2)] P = 0`.
    /// Verified with central finite differences: this is an independent check of
    /// the recurrence against the defining differential equation.
    #[test]
    fn legendre_satisfies_differential_equation() {
        let h = 1e-5_f64;
        for l in 0..=5u32 {
            for m in 0..=l {
                for &x in &[-0.6, -0.1, 0.3, 0.75] {
                    let p = associated_legendre_cs(l, m, x);
                    let pp = (associated_legendre_cs(l, m, x + h) - 2.0 * p
                        + associated_legendre_cs(l, m, x - h))
                        / (h * h);
                    let dp = (associated_legendre_cs(l, m, x + h)
                        - associated_legendre_cs(l, m, x - h))
                        / (2.0 * h);
                    let lf = l as f64;
                    let mf = m as f64;
                    let residual = (1.0 - x * x) * pp - 2.0 * x * dp
                        + (lf * (lf + 1.0) - mf * mf / (1.0 - x * x)) * p;
                    let scale = p.abs().max(1e-3);
                    assert!(
                        (residual / scale).abs() < 1e-4,
                        "l={l} m={m} x={x} residual={residual}"
                    );
                }
            }
        }
    }

    /// The Laguerre polynomials satisfy `x L'' + (a+1-x) L' + k L = 0`.
    ///
    /// The step size is chosen deliberately. A central second difference has
    /// truncation error `O(h^2)` and round-off error `O(eps/h^2)`, so the total
    /// error is minimised at `h ~ eps^(1/4) ~ 4e-4` for `f64`. Using `h = 1e-5`
    /// would amplify round-off by `1/h^2 = 1e10` and swamp the truncation term;
    /// this test uses the near-optimal step and documents the resulting budget.
    #[test]
    fn laguerre_satisfies_differential_equation() {
        let h = 4e-4_f64;
        for k in 0..=6u32 {
            for &a in &[1.0_f64, 3.0, 5.0, 7.0] {
                for &x in &[0.5, 2.0, 7.5, 20.0] {
                    let l = generalized_laguerre(k, a, x);
                    let lp = (generalized_laguerre(k, a, x + h)
                        - generalized_laguerre(k, a, x - h))
                        / (2.0 * h);
                    let lpp = (generalized_laguerre(k, a, x + h) - 2.0 * l
                        + generalized_laguerre(k, a, x - h))
                        / (h * h);
                    let residual = x * lpp + (a + 1.0 - x) * lp + k as f64 * l;
                    let scale = l.abs().max(1.0);
                    assert!(
                        (residual / scale).abs() < 1e-5,
                        "k={k} a={a} x={x} relative residual {}",
                        (residual / scale).abs()
                    );
                }
            }
        }
    }
}
