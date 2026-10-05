//! Évaluation : Chamfer (moyenne), Hausdorff (maximum), écart des normales, volume et aire.
use crate::{bvh::Bvh, mesh::*, pipeline::Model};
use rayon::prelude::*;

/// Distances en % de la diagonale de la boîte englobante de la référence.
#[derive(Clone, Copy, Debug, Default)]
pub struct Metrics { pub chamfer: f64, pub hausdorff: f64, pub normal_deg: f64, pub dvol: f64, pub darea: f64 }

pub fn dists(bvh: &Bvh, pts: &[V3]) -> Vec<(f64, u32)> {
    pts.par_iter().map(|&p| { let (d, f) = bvh.closest(p, f64::MAX); (d.sqrt(), f) }).collect()
}

/// Compare une reconstruction à la référence du modèle ; renvoie aussi l'écart signé par sommet (% diag.).
pub fn compare(rec: &Mesh, md: &Model) -> (Metrics, Vec<f32>) {
    if rec.f.is_empty() { return (Metrics { chamfer: f64::NAN, hausdorff: f64::NAN, ..Default::default() }, vec![]); }
    let (rm, rb) = md.reference();
    let k = 100.0 / md.diag_len;
    let fwd = dists(rb, &rec.v);
    let back = dists(&Bvh::new(rec), &md.samples);
    let mean = |d: &[(f64, u32)]| d.iter().map(|x| x.0).sum::<f64>() / d.len().max(1) as f64;
    let max = |d: &[(f64, u32)]| d.iter().map(|x| x.0).fold(0.0, f64::max);
    let vn = rec.vertex_normals();
    let ang: f64 = fwd.iter().zip(&vn).map(|(d, n)| {
        let r = unit(rm.fnormal(d.1 as usize));
        dot(r, n.map(|c| c as f64)).clamp(-1.0, 1.0).acos().to_degrees()
    }).sum::<f64>() / fwd.len() as f64;
    let rel = |a: f64, b: f64| 100.0 * (a - b) / b;
    (Metrics {
        chamfer: 0.5 * (mean(&fwd) + mean(&back)) * k, hausdorff: max(&fwd).max(max(&back)) * k, normal_deg: ang,
        dvol: rel(rec.volume(), rm.volume()), darea: rel(rec.area(), rm.area()),
    }, fwd.iter().zip(&rec.v).map(|(d, p)| {
        // Écart signé : négatif si le sommet est à l'intérieur de la référence, positif à l'extérieur.
        let s = dot(sub(*p, rm.tri(d.1 as usize)[0]), rm.fnormal(d.1 as usize));
        (d.0 * k * if s < 0.0 { -1.0 } else { 1.0 }) as f32
    }).collect())
}
