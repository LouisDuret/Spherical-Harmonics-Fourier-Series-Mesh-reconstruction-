//! Chaîne complète : import → diagnostic → réparation → décomposition → évaluation.
use crate::{bvh::Bvh, diag::*, mesh::*, methods::*, metrics::*, repair, sh::*};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Progression et annulation coopérative, partagées avec le thread appelant.
#[derive(Default)]
pub struct Ctl { pub cancel: AtomicBool, pub state: Mutex<(f32, String)> }
impl Ctl {
    pub fn step(&self, p: f32, msg: &str) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) { return Err("Calcul annulé".into()); }
        *self.state.lock().unwrap() = (p, msg.to_string());
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Method { A, B, C }
impl Method {
    pub const ALL: [Method; 3] = [Method::A, Method::B, Method::C];
    pub fn name(self) -> &'static str { match self { Method::A => "A · radiale r(θ,φ)", Method::B => "B · SPHARM paramétrique", Method::C => "C · implicite (Fourier 3D)" } }
    pub fn letter(self) -> &'static str { match self { Method::A => "A", Method::B => "B", Method::C => "C" } }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Params {
    pub method: Method,
    /// Degré maximal L (A, B) ou fréquence maximale K (C).
    pub lmax: usize,
    pub lambda: f64,
    pub weight: Weight,
    pub window: Window,
    pub area_fix: bool,
    pub area_passes: usize,
    pub align: bool,
    pub cmcf_iters: usize,
    pub cmcf_delta: f64,
    /// Taille de la grille de distance signée (méthode C).
    pub grid: usize,
}
impl Default for Params {
    fn default() -> Self { Params { method: Method::B, lmax: 40, lambda: 1e-8, weight: Weight::Area, window: Window::None, area_fix: true, area_passes: 100, align: true, cmcf_iters: 60, cmcf_delta: 3e-3, grid: 128 } }
}

#[derive(Clone, Debug)]
pub struct RepairParams { pub voxels: usize, pub closing: usize, pub tris: usize }
impl Default for RepairParams { fn default() -> Self { RepairParams { voxels: 160, closing: 2, tris: 10_000 } } }

/// Mesh de travail normalisé (sphère unité), sa référence pleine résolution et son diagnostic.
pub struct Model {
    pub name: String,
    pub mesh: Mesh,
    pub bvh: Bvh,
    pub full: Option<(Mesh, Bvh)>,
    pub xf: Xf,
    pub diag: Diag,
    pub diag_len: f64,
    pub samples: Vec<V3>,
    pub repaired: bool,
}

impl Model {
    /// `mesh` (et `full`) déjà normalisés ; `xf` ramène au repère d'origine.
    pub fn build(name: String, mut mesh: Mesh, xf: Xf, full: Option<Mesh>) -> Model {
        let mut bvh = Bvh::new(&mesh);
        let mut diag = diagnose(&mesh, &bvh);
        if diag.oriented && diag.watertight() && mesh.volume() < 0.0 {
            mesh.f.iter_mut().for_each(|t| t.swap(1, 2));
            bvh = Bvh::new(&mesh);
            diag = diagnose(&mesh, &bvh);
        }
        let r = full.as_ref().unwrap_or(&mesh);
        let step = r.v.len() / 30_000 + 1;
        let samples = r.v.iter().step_by(step).copied().collect();
        let diag_len = r.diagonal();
        let repaired = full.is_some();
        Model { name, bvh, full: full.map(|m| { let b = Bvh::new(&m); (m, b) }), mesh, xf, diag, diag_len, samples, repaired }
    }
    pub fn load(path: &Path) -> Result<Model, String> {
        let mut m = Mesh::load(path)?;
        let xf = m.normalize();
        Ok(Model::build(path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(), m, xf, None))
    }
    /// Surface de référence pour les mesures d'erreur.
    pub fn reference(&self) -> (&Mesh, &Bvh) { match &self.full { Some((m, b)) => (m, b), None => (&self.mesh, &self.bvh) } }

    /// Étape 3 : remaillage voxel étanche, puis simplification ; la pleine résolution sert de référence.
    pub fn repair(&self, rp: &RepairParams, ctl: &Ctl) -> Result<Model, String> {
        let mut full = repair::voxel_remesh(&self.mesh, rp.voxels, rp.closing, ctl)?;
        if full.f.is_empty() { return Err("réparation : iso-surface vide".into()); }
        let x2 = full.normalize();
        ctl.step(0.8, &format!("Simplification de {} à {} triangles", full.f.len(), rp.tris))?;
        let work = repair::simplify(&full, rp.tris);
        ctl.step(0.95, "Diagnostic")?;
        Ok(Model::build(self.name.clone(), work, self.xf.then(x2), Some(full)))
    }
}

pub struct ParamInfo { pub sphere: Vec<V3>, pub distortion: Vec<f64>, pub flipped: usize, pub p1: f64, pub dmax: f64, pub rot: [[f64; 3]; 3] }
pub enum Fit { Sh(ShFit), Fourier(FitC) }

pub struct Outcome {
    pub method: Method,
    pub lmax: usize,
    pub fit: Fit,
    pub energy: Vec<f64>,
    pub timings: Timings,
    pub warnings: Vec<String>,
    pub param: Option<ParamInfo>,
    pub metrics: Metrics,
}

impl Outcome {
    /// Reconstruction au degré `l` (coordonnées normalisées).
    pub fn mesh_at(&self, l: usize, win: Window, only: Option<usize>) -> Mesh {
        match &self.fit { Fit::Sh(f) => Mesh { v: f.positions(l, win, only), f: f.faces.clone() }, Fit::Fourier(f) => f.mesh(l, win) }
    }
    /// Nombre de coefficients réels stockés pour une troncature au degré `l`.
    pub fn ncoef(&self, l: usize) -> usize { match &self.fit { Fit::Sh(f) => (l + 1) * (l + 1) * f.nc, Fit::Fourier(_) => (2 * l + 1).pow(3) } }
}

pub fn compute(md: &Model, p: &Params, ctl: &Ctl) -> Result<Outcome, String> {
    let mut warnings: Vec<String> = md.diag.avail(p.method).map_err(|e| format!("Méthode {} indisponible ({e})", p.method.letter()))?.into_iter().collect();
    let (mut tm, mut param, mut lmax) = (Timings::new(), None, p.lmax);
    let fit = match p.method {
        Method::A => {
            let (fit, multi, missed) = fit_a(&md.bvh, lmax, ctl, &mut tm)?;
            if multi > 0.0 { warnings.push(format!("{:.1} % des directions coupent la surface plusieurs fois : seule l'enveloppe extérieure est gardée", 100.0 * multi)); }
            if missed > 0 { warnings.push(format!("{missed} rayons sans intersection : le centre est hors de l'objet")); }
            Fit::Sh(fit)
        }
        Method::B => {
            let t = Instant::now();
            let mut s = cmcf(&md.mesh, p.cmcf_iters, p.cmcf_delta, ctl)?;
            tm.push((format!("Paramétrage cMCF ({} it.)", p.cmcf_iters), t.elapsed().as_secs_f64()));
            if p.area_fix {
                let t = Instant::now();
                area_equalize(&mut s, &md.mesh, p.area_passes, ctl)?;
                tm.push(("Correction d'aire".into(), t.elapsed().as_secs_f64()));
            }
            let rot = if p.align { align_param(&md.mesh, &mut s, ctl)? } else { [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] };
            let (distortion, flipped) = param_quality(&md.mesh, &s);
            let mut d = distortion.clone();
            d.sort_by(f64::total_cmp);
            let (p1, dmax) = (1.0 / d[(d.len() * 99 / 100).min(d.len() - 1)], d[d.len() - 1]);
            if flipped > 0 { warnings.push(format!("{flipped} triangles inversés sur la sphère : paramétrage non bijectif")); }
            if p1 < 0.05 && !p.area_fix { warnings.push(format!("Zones fortement comprimées sur la sphère (jusqu'à ×{dmax:.0}) : envisager une correction d'aire")); }
            let t = Instant::now();
            let fit = fit_b(&md.mesh, &s, lmax, p.lambda, p.weight, ctl)?;
            tm.push((format!("Ajustement L = {lmax}"), t.elapsed().as_secs_f64()));
            param = Some(ParamInfo { sphere: s, distortion, flipped, p1, dmax, rot });
            Fit::Sh(fit)
        }
        Method::C => {
            let fit = fit_c(&md.bvh, p.grid.max(16), lmax, ctl, &mut tm)?;
            lmax = fit.kmax;
            Fit::Fourier(fit)
        }
    };
    let energy = match &fit { Fit::Sh(f) => energy(&f.coeffs, f.nc, f.lmax), Fit::Fourier(f) => f.energy() };
    let mut o = Outcome { method: p.method, lmax, fit, energy, timings: tm, warnings, param, metrics: Metrics::default() };
    ctl.step(0.92, "Mesures d'erreur")?;
    let t = Instant::now();
    o.metrics = compare(&o.mesh_at(lmax, p.window, None), md).0;
    o.timings.push(("Reconstruction + mesures".into(), t.elapsed().as_secs_f64()));
    Ok(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(m: Mesh) -> Model { let mut m = m; m.weld(); let xf = m.normalize(); Model::build("test".into(), m, xf, None) }

    #[test]
    fn base_orthonormee() {
        let sh = Sh::new(12);
        let (x, w) = gauss_legendre(16);
        let (pts, wv): (Vec<_>, Vec<_>) = (0..16 * 32).map(|i| ((x[i / 32], std::f64::consts::TAU * (i % 32) as f64 / 32.0), w[i / 32] * std::f64::consts::TAU / 32.0)).unzip();
        for probe in [0, 5, 37, 168] {
            let mut c = vec![0.0; sh.n()];
            c[probe] = 1.0;
            let v: Vec<f64> = synth(&sh, &pts, &c, 1).iter().zip(&wv).map(|(a, b)| a * b).collect();
            let g = adjoint(&sh, &pts, &v, 1);
            for (i, gi) in g.iter().enumerate() { assert!((gi - if i == probe { 1.0 } else { 0.0 }).abs() < 1e-12, "Gram[{probe}][{i}] = {gi}"); }
        }
    }

    #[test]
    fn formule_developpee_egale_recurrences() {
        let sh = Sh::new(9);
        let (th, ph, q) = (1.1f64, 0.7f64, sh.polys());
        let mut y = vec![0.0; sh.n()];
        sh.eval(th.cos(), ph, &mut y);
        for l in 0..=9usize { for m in -(l as i64)..=l as i64 {
            let am = m.unsigned_abs() as usize;
            let p: f64 = q[l * (l + 1) / 2 + am].iter().enumerate().map(|(e, c)| c * th.cos().powi(e as i32)).sum::<f64>() * th.sin().powi(am as i32);
            let v = if m == 0 { p } else { 2f64.sqrt() * p * if m > 0 { (am as f64 * ph).cos() } else { (am as f64 * ph).sin() } };
            assert!((v - y[(l * l) + (m + l as i64) as usize]).abs() < 1e-10, "Y[{l},{m}]");
        } }
    }

    #[test]
    fn formule_fourier_egale_champ() {
        let md = model(Mesh::torus(1.0, 0.35, 48, 24));
        let o = compute(&md, &Params { method: Method::C, lmax: 5, grid: 32, window: Window::Lanczos, ..Default::default() }, &Ctl::default()).unwrap();
        let Fit::Fourier(f) = &o.fit else { panic!() };
        let (field, u) = (f.field(5, Window::Lanczos), [7.0 / 32.0, 20.0 / 32.0, 13.0 / 32.0]);
        let sum: f64 = crate::export::rows(&o, 5, Window::Lanczos).iter().filter(|(k, _)| k.iter().find(|x| **x != 0).is_none_or(|x| *x > 0)).map(|(k, v)| {
            let a = std::f64::consts::TAU * (0..3).map(|i| k[i] as f64 * u[i]).sum::<f64>();
            if k.iter().all(|x| *x == 0) { v[0] } else { 2.0 * (v[0] * a.cos() - v[1] * a.sin()) }
        }).sum();
        assert!((sum - field[(7 * 32 + 20) * 32 + 13] as f64).abs() < 1e-5, "{sum}");
    }

    #[test]
    fn sphere_ne_donne_que_c00() {
        let md = model(Mesh::icosphere(4));
        let o = compute(&md, &Params { method: Method::A, lmax: 8, ..Default::default() }, &Ctl::default()).unwrap();
        let Fit::Sh(f) = &o.fit else { panic!() };
        assert!((f.coeffs[0] - 2.0 * std::f64::consts::PI.sqrt()).abs() < 5e-3, "c00 = {}", f.coeffs[0]);
        assert!(f.coeffs[1..].iter().all(|c| c.abs() < 5e-3));
        assert!(o.metrics.hausdorff < 0.5);
    }

    #[test]
    fn ellipsoide_methode_b_degre_1() {
        let s = Mesh::icosphere(3);
        let mut m = s.clone();
        m.v.iter_mut().for_each(|p| *p = [p[0] * 1.0, p[1] * 0.7, p[2] * 0.4]);
        let f = fit_b(&m, &s.v, 6, 0.0, Weight::Area, &Ctl::default()).unwrap();
        let e = energy(&f.coeffs, 3, 6);
        assert!(e[1] > 1.0 && e[0] < 1e-10 && e[2..].iter().all(|x| *x < 1e-10), "{e:?}");
    }

    #[test]
    fn ellipsoide_chaine_b_complete() {
        let mut m = Mesh::icosphere(4);
        m.v.iter_mut().for_each(|p| *p = [p[0] * 1.0, p[1] * 0.7, p[2] * 0.5]);
        let md = model(m);
        assert_eq!(md.diag.genus(), Some(0));
        let o = compute(&md, &Params { method: Method::B, lmax: 12, ..Default::default() }, &Ctl::default()).unwrap();
        assert_eq!(o.param.as_ref().unwrap().flipped, 0);
        assert!(o.metrics.hausdorff < 1.0, "{:?}", o.metrics);
    }

    #[test]
    fn tore_refuse_par_b_accepte_par_c() {
        let md = model(Mesh::torus(1.0, 0.35, 48, 24));
        assert_eq!(md.diag.genus(), Some(1));
        assert!(compute(&md, &Params { method: Method::B, ..Default::default() }, &Ctl::default()).is_err());
        let o = compute(&md, &Params { method: Method::C, lmax: 12, grid: 64, window: Window::Lanczos, ..Default::default() }, &Ctl::default()).unwrap();
        assert!(o.metrics.chamfer < 1.0, "{:?}", o.metrics);
    }

    #[test]
    fn reparation_surface_ouverte() {
        let mut m = Mesh::icosphere(3);
        m.f.truncate(m.f.len() - 3);
        let md = model(m);
        assert!(!md.diag.watertight());
        let r = md.repair(&RepairParams { voxels: 48, closing: 2, tris: 2000 }, &Ctl::default()).unwrap();
        assert!(r.diag.watertight() && r.diag.genus() == Some(0), "{}", r.diag.summary());
    }

    /// Trou bien plus large que la fermeture (cas du Stanford Bunny) : le volume doit rester plein.
    #[test]
    fn reparation_grand_trou() {
        let mut m = Mesh::icosphere(4);
        let f = std::mem::take(&mut m.f);
        m.f = f.into_iter().filter(|t| t.iter().any(|&i| m.v[i as usize][2] > -0.8)).collect();
        let r = model(m).repair(&RepairParams { voxels: 48, closing: 2, tris: 2000 }, &Ctl::default()).unwrap();
        assert!(r.diag.watertight() && r.diag.genus() == Some(0), "{}", r.diag.summary());
        let (v, full) = (r.mesh.volume(), 4.0 / 3.0 * std::f64::consts::PI * (r.diag_len / 12f64.sqrt()).powi(3));
        assert!(v > 0.8 * full, "volume {v} / {full}");
    }
}
