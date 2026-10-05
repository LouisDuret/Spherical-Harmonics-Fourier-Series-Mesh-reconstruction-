//! Les trois décompositions : A radiale r(θ,φ), B paramétrique SPHARM, C implicite (Fourier 3D).
use crate::{bvh::Bvh, mesh::*, pipeline::Ctl, repair::marching_tets, sh::*};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::{PI, TAU};
use std::time::Instant;

pub type Timings = Vec<(String, f64)>;

/// Résultat des méthodes A (nc = 1) et B (nc = 3), avec les contributions par degré aux sommets d'affichage.
pub struct ShFit {
    pub lmax: usize, pub nc: usize,
    pub coeffs: Vec<f64>,
    pub faces: Vec<[u32; 3]>,
    pub dirs: Vec<V3>,
    pub terms: Vec<f32>,
    pub n: usize,
}

impl ShFit {
    /// Sommets reconstruits au degré `l` (somme partielle). `only` isole la contribution d'un degré.
    pub fn positions(&self, l: usize, win: Window, only: Option<usize>) -> Vec<V3> {
        let (nl, nc, l) = (self.lmax + 1, self.nc, l.min(self.lmax));
        let base = if nc == 1 { 0 } else { 1 };
        (0..self.n).into_par_iter().map(|i| {
            let t = &self.terms[i * nl * nc..(i + 1) * nl * nc];
            let mut s = [0.0; 3];
            for d in 0..=if only.is_some() { self.lmax } else { l } {
                let g = match only { Some(o) => if d <= base || d == o { 1.0 } else { 0.0 }, None => win.sigma(d as f64, l) };
                for c in 0..nc { s[c] += g * t[d * nc + c] as f64; }
            }
            if nc == 1 { mul(self.dirs[i], s[0]) } else { s }
        }).collect()
    }
}

/// Grille latitude-longitude fermée par deux pôles : directions, (cos θ, φ), triangles orientés vers l'extérieur.
pub fn latlon_grid(nlat: usize, nlon: usize) -> (Vec<V3>, Vec<(f64, f64)>, Vec<[u32; 3]>) {
    let (mut dirs, mut pts, mut f) = (vec![], vec![], vec![]);
    for j in 0..nlat { for k in 0..nlon {
        let (th, ph) = (PI * (j as f64 + 0.5) / nlat as f64, TAU * k as f64 / nlon as f64);
        dirs.push([th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()]);
        pts.push((th.cos(), ph));
    } }
    dirs.extend([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]);
    pts.extend([(1.0, 0.0), (-1.0, 0.0)]);
    let id = |j: usize, k: usize| (j * nlon + k % nlon) as u32;
    let (np, sp) = ((nlat * nlon) as u32, (nlat * nlon + 1) as u32);
    for k in 0..nlon {
        f.push([np, id(0, k), id(0, k + 1)]);
        for j in 0..nlat - 1 { f.push([id(j, k), id(j + 1, k), id(j + 1, k + 1)]); f.push([id(j, k), id(j + 1, k + 1), id(j, k + 1)]); }
        f.push([id(nlat - 1, k + 1), id(nlat - 1, k), sp]);
    }
    (dirs, pts, f)
}

/// Méthode A : r(θ,φ) par lancer de rayons sur une grille de Gauss-Legendre, puis quadrature exacte.
/// Renvoie aussi la proportion de directions non étoilées et le nombre de rayons sans intersection.
pub fn fit_a(bvh: &Bvh, lmax: usize, ctl: &Ctl, tm: &mut Timings) -> Result<(ShFit, f64, usize), String> {
    ctl.step(0.1, "Lancer de rayons")?;
    let t = Instant::now();
    let nlat = (lmax + 1).max(128);
    let nlon = 2 * nlat;
    let (x, w) = gauss_legendre(nlat);
    let pts: Vec<(f64, f64)> = (0..nlat * nlon).map(|i| (x[i / nlon], TAU * (i % nlon) as f64 / nlon as f64)).collect();
    let rays: Vec<(f64, usize)> = pts.par_iter().map_init(Vec::new, |hits, &(ct, ph)| {
        let st = (1.0 - ct * ct).sqrt();
        bvh.ray([0.0; 3], [st * ph.cos(), st * ph.sin(), ct], hits);
        (hits.last().map_or(0.0, |h| h.0), hits.len())
    }).collect();
    let multi = rays.iter().filter(|r| r.1 > 1).count() as f64 / rays.len() as f64;
    let missed = rays.iter().filter(|r| r.1 == 0).count();
    tm.push((format!("Lancer de {} rayons", rays.len()), t.elapsed().as_secs_f64()));
    ctl.step(0.5, "Transformée en harmoniques")?;
    let t = Instant::now();
    let sh = Sh::new(lmax);
    let wv: Vec<f64> = rays.iter().enumerate().map(|(i, r)| r.0 * w[i / nlon] * TAU / nlon as f64).collect();
    let coeffs = adjoint(&sh, &pts, &wv, 1);
    tm.push(("Transformée (quadrature)".into(), t.elapsed().as_secs_f64()));
    ctl.step(0.8, "Contributions par degré")?;
    let (dirs, dp, faces) = latlon_grid(96, 192);
    let terms = degree_terms(&sh, &dp, &coeffs, 1);
    Ok((ShFit { lmax, nc: 1, coeffs, faces, n: dirs.len(), dirs, terms }, multi, missed))
}

/// Gradient conjugué préconditionné, sans matrice. `x` contient l'estimation initiale.
fn pcg(apply: impl Fn(&[f64]) -> Vec<f64>, pre: impl Fn(usize) -> f64, b: &[f64], x: &mut [f64], tol: f64, maxit: usize, mut tick: impl FnMut(usize) -> Result<(), String>) -> Result<usize, String> {
    let dotv = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let ax = apply(x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let mut z: Vec<f64> = r.iter().enumerate().map(|(i, v)| v * pre(i)).collect();
    let mut p = z.clone();
    let (mut rz, lim) = (dotv(&r, &z), tol * tol * dotv(b, b));
    for it in 0..maxit {
        if dotv(&r, &r) <= lim { return Ok(it); }
        tick(it)?;
        let hp = apply(&p);
        let alpha = rz / dotv(&p, &hp);
        for i in 0..x.len() { x[i] += alpha * p[i]; r[i] -= alpha * hp[i]; z[i] = r[i] * pre(i); }
        let rz2 = dotv(&r, &z);
        for i in 0..p.len() { p[i] = z[i] + rz2 / rz * p[i]; }
        rz = rz2;
    }
    Ok(maxit)
}

struct Csr { ptr: Vec<usize>, col: Vec<u32>, val: Vec<f64>, diag: Vec<f64> }

fn cot_laplacian(x: &[V3], f: &[[u32; 3]]) -> Csr {
    let mut rows: Vec<Vec<(u32, f64)>> = vec![vec![]; x.len()];
    for t in f { for k in 0..3 {
        let (i, j, o) = (t[k] as usize, t[(k + 1) % 3] as usize, t[(k + 2) % 3] as usize);
        let (u, w) = (sub(x[i], x[o]), sub(x[j], x[o]));
        let cot = 0.5 * (dot(u, w) / norm(cross(u, w)).max(1e-12)).clamp(-5.0, 50.0);
        rows[i].extend([(j as u32, cot), (i as u32, -cot)]);
        rows[j].extend([(i as u32, cot), (j as u32, -cot)]);
    } }
    let mut c = Csr { ptr: vec![0], col: vec![], val: vec![], diag: vec![0.0; x.len()] };
    for (i, mut r) in rows.into_iter().enumerate() {
        r.sort_by_key(|e| e.0);
        let start = c.col.len();
        for (j, v) in r {
            if j as usize == i { c.diag[i] += v; }
            if c.col.len() > start && *c.col.last().unwrap() == j { *c.val.last_mut().unwrap() += v; } else { c.col.push(j); c.val.push(v); }
        }
        c.ptr.push(c.col.len());
    }
    c
}

fn sphere_normalise(x: &mut [V3], f: &[[u32; 3]]) {
    let (mut c, mut a) = ([0.0; 3], 0.0);
    for t in f {
        let p = t.map(|i| x[i as usize]);
        let w = norm(cross(sub(p[1], p[0]), sub(p[2], p[0]))) * 0.5;
        c = add(c, mul(add(add(p[0], p[1]), p[2]), w / 3.0));
        a += w;
    }
    let (c, s) = (mul(c, 1.0 / a), (4.0 * PI / a).sqrt());
    for p in x { *p = mul(sub(*p, c), s); }
}

/// Paramétrage sphérique par flot de courbure moyenne conformalisé (Kazhdan et al. 2012).
pub fn cmcf(m: &Mesh, iters: usize, delta: f64, ctl: &Ctl) -> Result<Vec<V3>, String> {
    let (f, n) = (&m.f, m.v.len());
    let mut x = m.v.clone();
    sphere_normalise(&mut x, f);
    let lap = cot_laplacian(&x, f);
    for it in 0..iters {
        ctl.step(0.05 + 0.4 * it as f32 / iters as f32, &format!("Paramétrage cMCF {}/{iters}", it + 1))?;
        let mass: Vec<f64> = vertex_areas(&x, f).into_iter().map(|a| a + 1e-12).collect();
        let apply = |u: &[f64]| -> Vec<f64> {
            (0..n).into_par_iter().map(|i| mass[i] * u[i] - delta * (lap.ptr[i]..lap.ptr[i + 1]).map(|k| lap.val[k] * u[lap.col[k] as usize]).sum::<f64>()).collect()
        };
        for c in 0..3 {
            let b: Vec<f64> = (0..n).map(|i| mass[i] * x[i][c]).collect();
            let mut u: Vec<f64> = x.iter().map(|p| p[c]).collect();
            pcg(&apply, |i| 1.0 / (mass[i] - delta * lap.diag[i]), &b, &mut u, 1e-9, 3000, |_| Ok(()))?;
            for i in 0..n { x[i][c] = u[i]; }
        }
        sphere_normalise(&mut x, f);
    }
    Ok(x.into_iter().map(unit).collect())
}

fn det3(a: V3, b: V3, c: V3) -> f64 { dot(a, cross(b, c)) }

/// Correction de la distorsion d'aire par égalisation de densité « en tranches » : pour une suite d'axes,
/// on remappe la coordonnée z = s·axe par la fonction de répartition (lissée) de l'aire d'origine.
/// Chaque passe est un difféomorphisme monotone de la sphère ; une passe qui inverserait un triangle est amortie ou rejetée.
pub fn area_equalize(s: &mut [V3], m: &Mesh, passes: usize, ctl: &Ctl) -> Result<(), String> {
    const B: usize = 48;
    let mass = m.vertex_areas();
    let total: f64 = mass.iter().sum();
    let flipped = |s: &[V3]| m.f.iter().any(|t| det3(s[t[0] as usize], s[t[1] as usize], s[t[2] as usize]) <= 1e-14);
    for it in 0..passes {
        ctl.step(0.45 + 0.1 * it as f32 / passes as f32, &format!("Correction d'aire {}/{passes}", it + 1))?;
        let a = crate::diag::fibonacci_dir(it * 197 % 509, 509);
        let mut h = [2e-3 * total / B as f64; B];
        for (p, w) in s.iter().zip(&mass) {
            let x = ((dot(*p, a) + 1.0) * 0.5 * B as f64 - 0.5).clamp(0.0, B as f64 - 1.0);
            let (k, f) = (x.floor() as usize, x.fract());
            h[k] += w * (1.0 - f);
            h[(k + 1).min(B - 1)] += w * f;
        }
        let mut cdf = [0.0; B + 1];
        for k in 0..B { cdf[k + 1] = cdf[k] + h[k]; }
        let map = |z: f64| { let u = ((z + 1.0) * 0.5 * B as f64).clamp(0.0, B as f64 - 1e-9); let k = u.floor() as usize; 2.0 * (cdf[k] + (u - k as f64) * h[k]) / cdf[B] - 1.0 };
        let mut alpha = 0.5;
        while alpha > 0.02 {
            let cand: Vec<V3> = s.iter().map(|&p| {
                let z = dot(p, a);
                let z2 = (z + alpha * (map(z) - z)).clamp(-1.0, 1.0);
                if 1.0 - z * z < 1e-12 { return p; }
                unit(add(mul(a, z2), mul(sub(p, mul(a, z)), ((1.0 - z2 * z2) / (1.0 - z * z)).sqrt())))
            }).collect();
            if !flipped(&cand) { s.copy_from_slice(&cand); break; }
            alpha *= 0.5;
        }
    }
    Ok(())
}

/// Distorsion d'aire par sommet (part d'aire d'origine / part d'aire sur la sphère) et nombre de triangles inversés.
pub fn param_quality(m: &Mesh, s: &[V3]) -> (Vec<f64>, usize) {
    let (a0, a1) = (m.vertex_areas(), vertex_areas(s, &m.f));
    let (s0, s1): (f64, f64) = (a0.iter().sum(), a1.iter().sum());
    let dist = a0.iter().zip(&a1).map(|(a, b)| (a / s0) / (b / s1).max(1e-300)).collect();
    let flipped = m.f.iter().filter(|t| det3(s[t[0] as usize], s[t[1] as usize], s[t[2] as usize]) <= 0.0).count();
    (dist, flipped)
}

fn eig3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..60 { for (p, q) in [(0, 1), (0, 2), (1, 2)] {
        if a[p][q].abs() < 1e-300 { continue; }
        let (s, c) = (0.5 * (2.0 * a[p][q]).atan2(a[q][q] - a[p][p])).sin_cos();
        for k in 0..3 { let (x, y) = (a[k][p], a[k][q]); a[k][p] = c * x - s * y; a[k][q] = s * x + c * y; }
        for k in 0..3 { let (x, y) = (a[p][k], a[q][k]); a[p][k] = c * x - s * y; a[q][k] = s * x + c * y; }
        for k in 0..3 { let (x, y) = (v[k][p], v[k][q]); v[k][p] = c * x - s * y; v[k][q] = s * x + c * y; }
    } }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Alignement du paramétrage (convention SPHARM) : rotation de la sphère qui aligne l'ellipsoïde du degré 1
/// (grand axe → x, petit axe → z). Renvoie la matrice de rotation appliquée (lignes = nouveaux axes).
pub fn align_param(m: &Mesh, s: &mut [V3], ctl: &Ctl) -> Result<[[f64; 3]; 3], String> {
    let c = fit_b(m, s, 1, 0.0, Weight::Area, ctl)?.coeffs;
    let col = |i: usize, sg: f64| [c[i * 3] * sg, c[i * 3 + 1] * sg, c[i * 3 + 2] * sg];
    let a = [col(3, -1.0), col(1, -1.0), col(2, 1.0)];
    let (ev, v) = eig3([0, 1, 2].map(|i| [0, 1, 2].map(|j| dot(a[i], a[j]))));
    let mut ord = [0, 1, 2];
    ord.sort_by(|&i, &j| ev[j].total_cmp(&ev[i]));
    let mut r = ord.map(|o| [v[0][o], v[1][o], v[2][o]]);
    if det3(r[0], r[1], r[2]) < 0.0 { r[2] = mul(r[2], -1.0); }
    for p in s.iter_mut() { *p = unit([dot(r[0], *p), dot(r[1], *p), dot(r[2], *p)]); }
    Ok(r)
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Weight { Area, Sphere, Uniform }
impl Weight { pub fn name(self) -> &'static str { match self { Weight::Area => "aire", Weight::Sphere => "aire sphère", Weight::Uniform => "uniforme" } } }

/// Méthode B : ajustement de x, y, z(θ,φ) par moindres carrés pondérés et régularisés,
/// résolu par gradient conjugué sans matrice (synthèse + adjoint), préconditionné par degré.
pub fn fit_b(m: &Mesh, s: &[V3], lmax: usize, lam: f64, weight: Weight, ctl: &Ctl) -> Result<ShFit, String> {
    let sh = Sh::new(lmax);
    let (n, nc) = (sh.n(), 3);
    let pts: Vec<(f64, f64)> = s.iter().map(|p| (p[2].clamp(-1.0, 1.0), p[1].atan2(p[0]))).collect();
    let mut w = match weight { Weight::Area => m.vertex_areas(), Weight::Sphere => vertex_areas(s, &m.f), Weight::Uniform => vec![1.0; s.len()] };
    let ws: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= ws);
    let reg: Vec<f64> = (0..n).map(|i| { let l = (i as f64).sqrt().floor(); lam * (l * (l + 1.0)).powi(2) }).collect();
    let wv: Vec<f64> = m.v.iter().zip(&w).flat_map(|(p, w)| p.map(|c| c * w)).collect();
    let b = adjoint(&sh, &pts, &wv, nc);
    let apply = |c: &[f64]| -> Vec<f64> {
        let mut y = synth(&sh, &pts, c, nc);
        y.chunks_mut(nc).zip(&w).for_each(|(v, w)| v.iter_mut().for_each(|x| *x *= w));
        let mut h = adjoint(&sh, &pts, &y, nc);
        for i in 0..n * nc { h[i] += reg[i / nc] * c[i]; }
        h
    };
    let mut coeffs = vec![0.0; n * nc];
    let its = pcg(apply, |i| 1.0 / (1.0 / (4.0 * PI) + reg[i / nc]), &b, &mut coeffs, 1e-7, 600, |it| ctl.step(0.6 + 0.3 * (it as f32 / 600.0).sqrt(), &format!("Ajustement SPHARM L = {lmax} (itération {it})")))?;
    let _ = its;
    let terms = degree_terms(&sh, &pts, &coeffs, nc);
    Ok(ShFit { lmax, nc, coeffs, faces: m.f.clone(), dirs: vec![], terms, n: s.len() })
}

/// Méthode C : coefficients de Fourier 3D de la distance signée, cube |k| ≤ kmax.
/// f(x) = Σ_k c_k exp(2πi k·u), u = (x + h) / 2h (coordonnées normalisées).
pub struct FitC { pub m: usize, pub h: f64, pub kmax: usize, pub c: Vec<Complex<f64>> }

/// FFT 3D en place : trois passes 1D sur l'axe contigu, séparées par une rotation des axes.
fn fft3(a: &mut Vec<Complex<f64>>, m: usize, inverse: bool) {
    let mut pl = FftPlanner::new();
    let fft = if inverse { pl.plan_fft_inverse(m) } else { pl.plan_fft_forward(m) };
    let mut b = vec![Complex::new(0.0, 0.0); a.len()];
    for _ in 0..3 {
        a.par_chunks_mut(m).for_each_init(|| vec![Complex::new(0.0, 0.0); fft.get_inplace_scratch_len()], |s, c| fft.process_with_scratch(c, s));
        b.par_chunks_mut(m * m).enumerate().for_each(|(k, slab)| for i in 0..m { for j in 0..m { slab[i * m + j] = a[(i * m + j) * m + k]; } });
        std::mem::swap(a, &mut b);
    }
}

/// Distance signée sur une grille m³ : distance exacte par BVH, signe par parité le long de rayons en z.
fn sdf_grid(bvh: &Bvh, m: usize, h: f64) -> Vec<f32> {
    let cell = 2.0 * h / m as f64;
    let mut out = vec![0f32; m * m * m];
    out.par_chunks_mut(m).enumerate().for_each_init(Vec::new, |hits: &mut Vec<(f64, u32)>, (c, col)| {
        let (x, y, z0) = (-h + (c / m) as f64 * cell, -h + (c % m) as f64 * cell, -h - 1.0);
        bvh.ray([x, y, z0], [0.0, 0.0, 1.0], hits);
        if hits.len() % 2 == 1 { bvh.ray([x + 1e-7, y + 1.3e-7, z0], [0.0, 0.0, 1.0], hits); }
        let (mut prev, mut nh) = (f64::MAX, 0);
        for (k, o) in col.iter_mut().enumerate() {
            let z = -h + k as f64 * cell;
            while nh < hits.len() && z0 + hits[nh].0 < z { nh += 1; }
            let d = bvh.closest([x, y, z], if prev == f64::MAX { prev } else { (prev + cell).powi(2) * 1.001 }).0.sqrt();
            prev = d;
            *o = (if nh % 2 == 1 { -d } else { d }) as f32;
        }
    });
    out
}

pub fn fit_c(bvh: &Bvh, m: usize, kmax: usize, ctl: &Ctl, tm: &mut Timings) -> Result<FitC, String> {
    let (h, kmax) = (1.25, kmax.min(m / 2 - 1));
    ctl.step(0.1, &format!("Distance signée sur grille {m}³"))?;
    let t = Instant::now();
    let sdf = sdf_grid(bvh, m, h);
    tm.push((format!("Distance signée {m}³"), t.elapsed().as_secs_f64()));
    ctl.step(0.6, "FFT 3D")?;
    let t = Instant::now();
    let mut a: Vec<Complex<f64>> = sdf.iter().map(|&v| Complex::new(v as f64, 0.0)).collect();
    fft3(&mut a, m, false);
    let (s, kk, inv) = (2 * kmax + 1, kmax as i64, 1.0 / (m * m * m) as f64);
    let w = |q: usize| (q as i64 - kk).rem_euclid(m as i64) as usize;
    let c = (0..s * s * s).map(|i| a[(w(i / (s * s)) * m + w(i / s % s)) * m + w(i % s)] * inv).collect();
    tm.push(("FFT 3D".into(), t.elapsed().as_secs_f64()));
    Ok(FitC { m, h, kmax, c })
}

impl FitC {
    /// (kx, ky, kz) du i-ème coefficient.
    pub fn k_of(&self, i: usize) -> [i64; 3] { let (s, kk) = (2 * self.kmax + 1, self.kmax as i64); [(i / (s * s)) as i64 - kk, (i / s % s) as i64 - kk, (i % s) as i64 - kk] }
    /// Facteur d'apodisation séparable pour une troncature à `k`.
    pub fn sigma(&self, i: usize, k: usize, win: Window) -> f64 {
        let q = self.k_of(i);
        if q.iter().any(|x| x.unsigned_abs() as usize > k) { 0.0 } else { q.iter().map(|&x| win.sigma(x.abs() as f64, k)).product() }
    }
    /// Série tronquée à |k| ≤ k, évaluée sur la grille m³.
    pub fn field(&self, k: usize, win: Window) -> Vec<f32> {
        let m = self.m;
        let mut a = vec![Complex::new(0.0, 0.0); m * m * m];
        let w = |q: i64| q.rem_euclid(m as i64) as usize;
        for (i, c) in self.c.iter().enumerate() {
            let (g, q) = (self.sigma(i, k.min(self.kmax), win), self.k_of(i));
            if g != 0.0 { a[(w(q[0]) * m + w(q[1])) * m + w(q[2])] = c * g; }
        }
        fft3(&mut a, m, true);
        a.iter().map(|z| z.re as f32).collect()
    }
    /// Surface {f = 0} de la série tronquée.
    pub fn mesh(&self, k: usize, win: Window) -> Mesh { marching_tets(&self.field(k, win), [self.m; 3], [-self.h; 3], 2.0 * self.h / self.m as f64) }
    /// Énergie par couche max|k_i| = s.
    pub fn energy(&self) -> Vec<f64> {
        let mut e = vec![0.0; self.kmax + 1];
        for (i, c) in self.c.iter().enumerate() { e[self.k_of(i).iter().map(|x| x.unsigned_abs() as usize).max().unwrap()] += c.norm_sqr(); }
        e
    }
}
