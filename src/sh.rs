//! Harmoniques sphériques réelles orthonormées (phase de Condon–Shortley, θ = colatitude).
//! Indice du coefficient (l, m) : l² + l + m. Tout est évalué sans matrice (synthèse / adjoint).
use rayon::prelude::*;
use std::f64::consts::{PI, SQRT_2};

pub struct Sh { pub lmax: usize, a: Vec<f64>, b: Vec<f64>, d: Vec<f64> }

impl Sh {
    pub fn new(lmax: usize) -> Sh {
        let t = |l: usize, m: usize| l * (l + 1) / 2 + m;
        let mut a = vec![0.0; t(lmax, lmax) + 1];
        let mut b = a.clone();
        for l in 1..=lmax { for m in 0..l {
            let (lf, mf) = (l as f64, m as f64);
            a[t(l, m)] = ((4.0 * lf * lf - 1.0) / (lf * lf - mf * mf)).sqrt();
            b[t(l, m)] = (((lf - 1.0).powi(2) - mf * mf) / (4.0 * (lf - 1.0).powi(2) - 1.0)).max(0.0).sqrt();
        } }
        let d = (0..=lmax).map(|m| if m == 0 { 1.0 } else { ((2 * m + 1) as f64 / (2 * m) as f64).sqrt() }).collect();
        Sh { lmax, a, b, d }
    }
    /// Forme développée : coefficients (puissances croissantes de x = cos θ) du polynôme Q tel que
    /// P̄_l^m(x) = (1 − x²)^(m/2) · Q(x) ; indice l(l+1)/2 + m.
    pub fn polys(&self) -> Vec<Vec<f64>> {
        let t = |l: usize, m: usize| l * (l + 1) / 2 + m;
        let mut out = vec![vec![]; t(self.lmax, self.lmax) + 1];
        let mut pmm = 0.28209479177387814;
        for m in 0..=self.lmax {
            if m > 0 { pmm *= -self.d[m]; }
            let (mut p1, mut p2): (Vec<f64>, Vec<f64>) = (vec![pmm], vec![]);
            out[t(m, m)] = p1.clone();
            for l in m + 1..=self.lmax {
                let (a, b) = (self.a[t(l, m)], self.b[t(l, m)]);
                let mut p = vec![0.0; l - m + 1];
                for (i, c) in p1.iter().enumerate() { p[i + 1] += a * c; }
                for (i, c) in p2.iter().enumerate() { p[i] -= a * b * c; }
                out[t(l, m)] = p.clone();
                p2 = p1;
                p1 = p;
            }
        }
        out
    }
    pub fn n(&self) -> usize { (self.lmax + 1) * (self.lmax + 1) }

    /// Toutes les Y_lm en (cos θ, φ), par récurrences stables sur les fonctions de Legendre normalisées.
    pub fn eval(&self, ct: f64, phi: f64, y: &mut [f64]) {
        let st = (1.0 - ct * ct).max(0.0).sqrt();
        let (s1, c1) = phi.sin_cos();
        let (mut cm, mut sm, mut pmm) = (1.0, 0.0, 0.28209479177387814);
        for m in 0..=self.lmax {
            if m > 0 { pmm *= -self.d[m] * st; let c = cm * c1 - sm * s1; sm = sm * c1 + cm * s1; cm = c; }
            let (mut p1, mut p2) = (pmm, 0.0);
            for l in m..=self.lmax {
                if l > m { let t = l * (l + 1) / 2 + m; let p = self.a[t] * (ct * p1 - self.b[t] * p2); p2 = p1; p1 = p; }
                let i = l * l + l;
                if m == 0 { y[i] = p1; } else { y[i + m] = SQRT_2 * p1 * cm; y[i - m] = SQRT_2 * p1 * sm; }
            }
        }
    }
}

/// Synthèse : valeurs (N × nc) de la série de coefficients `c` ((L+1)² × nc) aux points (cos θ, φ).
pub fn synth(sh: &Sh, pts: &[(f64, f64)], c: &[f64], nc: usize) -> Vec<f64> {
    let mut out = vec![0.0; pts.len() * nc];
    out.par_chunks_mut(nc).zip(pts.par_iter()).for_each_init(|| vec![0.0; sh.n()], |y, (o, &(ct, ph))| {
        sh.eval(ct, ph, y);
        for (i, yi) in y.iter().enumerate() { for k in 0..nc { o[k] += yi * c[i * nc + k]; } }
    });
    out
}

/// Adjoint de la synthèse : Σ_i Y(p_i) v_i. Avec des poids de quadrature, c'est l'analyse exacte.
pub fn adjoint(sh: &Sh, pts: &[(f64, f64)], v: &[f64], nc: usize) -> Vec<f64> {
    let n = sh.n();
    pts.par_chunks(64).zip(v.par_chunks(64 * nc)).fold(|| (vec![0.0; n], vec![0.0; n * nc]), |(mut y, mut acc), (p, val)| {
        for (j, &(ct, ph)) in p.iter().enumerate() {
            sh.eval(ct, ph, &mut y);
            for i in 0..n { for k in 0..nc { acc[i * nc + k] += y[i] * val[j * nc + k]; } }
        }
        (y, acc)
    }).map(|x| x.1).reduce(|| vec![0.0; n * nc], |mut a, b| { for (x, y) in a.iter_mut().zip(&b) { *x += y; } a })
}

/// Contribution de chaque degré en chaque point : tableau [point][l][composante] (curseur de degré en O(N)).
pub fn degree_terms(sh: &Sh, pts: &[(f64, f64)], c: &[f64], nc: usize) -> Vec<f32> {
    let nl = sh.lmax + 1;
    let mut out = vec![0f32; pts.len() * nl * nc];
    out.par_chunks_mut(nl * nc).zip(pts.par_iter()).for_each_init(|| vec![0.0; sh.n()], |y, (o, &(ct, ph))| {
        sh.eval(ct, ph, y);
        for l in 0..nl { for k in 0..nc { o[l * nc + k] = (l * l..(l + 1) * (l + 1)).map(|i| y[i] * c[i * nc + k]).sum::<f64>() as f32; } }
    });
    out
}

/// Spectre d'énergie par degré : E_l = Σ_m |c_lm|².
pub fn energy(c: &[f64], nc: usize, lmax: usize) -> Vec<f64> {
    (0..=lmax).map(|l| c[l * l * nc..(l + 1) * (l + 1) * nc].iter().map(|x| x * x).sum()).collect()
}

/// Nœuds (cos θ, décroissants) et poids de Gauss-Legendre.
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut w) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let (mut z, mut dp) = ((PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos(), 1.0);
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0, z);
            for k in 2..=n { let kf = k as f64; let p = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf; p0 = p1; p1 = p; }
            dp = n as f64 * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-15 { break; }
        }
        x[i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * dp * dp);
    }
    (x, w)
}

/// Fenêtre d'apodisation contre le phénomène de Gibbs.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Window { None, Lanczos, Hann }
impl Window {
    /// Facteur appliqué au degré `l` pour une troncature au degré `top`.
    pub fn sigma(self, l: f64, top: usize) -> f64 {
        let x = l / (top as f64 + 1.0);
        match self {
            Window::None => 1.0,
            Window::Lanczos => if x == 0.0 { 1.0 } else { (PI * x).sin() / (PI * x) },
            Window::Hann => 0.5 * (1.0 + (PI * x).cos()),
        }
    }
    pub fn name(self) -> &'static str { match self { Window::None => "aucune", Window::Lanczos => "Lanczos", Window::Hann => "Hann" } }
}
