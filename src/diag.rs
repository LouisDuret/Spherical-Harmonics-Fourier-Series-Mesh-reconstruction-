//! Diagnostic topologique : composantes, variété, étanchéité, genre, caractère étoilé.
use crate::{bvh::Bvh, mesh::*, pipeline::Method};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct Comp { pub nv: usize, pub ne: usize, pub nf: usize, pub boundary: usize, pub nonmanifold: usize, pub euler: i64 }
impl Comp {
    pub fn closed(&self) -> bool { self.boundary == 0 && self.nonmanifold == 0 }
    pub fn genus(&self) -> Option<i64> { self.closed().then_some((2 - self.euler) / 2) }
}

#[derive(Clone, Debug, Default)]
pub struct Diag {
    pub nv: usize, pub nf: usize,
    pub comps: Vec<Comp>,
    pub oriented: bool,
    /// Proportion de rayons issus du centre qui ne coupent la surface qu'une fois.
    pub star: Option<f64>,
}

pub fn fibonacci_dir(i: usize, n: usize) -> V3 {
    let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
    let (r, a) = ((1.0 - z * z).sqrt(), i as f64 * 2.399963229728653);
    [r * a.cos(), r * a.sin(), z]
}

pub fn diagnose(m: &Mesh, bvh: &Bvh) -> Diag {
    let (lab, nc) = m.components();
    let mut comps = vec![Comp::default(); nc];
    let mut seen = vec![false; m.v.len()];
    for (i, f) in m.f.iter().enumerate() {
        let c = &mut comps[lab[i] as usize];
        c.nf += 1;
        for &v in f { if !seen[v as usize] { seen[v as usize] = true; c.nv += 1; } }
    }
    let mut edges: HashMap<(u32, u32), (u32, i32, u32)> = HashMap::new();
    for (i, f) in m.f.iter().enumerate() { for k in 0..3 {
        let (a, b) = (f[k], f[(k + 1) % 3]);
        let e = edges.entry((a.min(b), a.max(b))).or_insert((0, 0, lab[i]));
        e.0 += 1;
        e.1 += if a < b { 1 } else { -1 };
    } }
    let mut oriented = true;
    for (n, bal, c) in edges.values() {
        let c = &mut comps[*c as usize];
        c.ne += 1;
        match n { 1 => c.boundary += 1, 2 => if *bal != 0 { oriented = false }, _ => c.nonmanifold += 1 }
    }
    for c in &mut comps { c.euler = c.nv as i64 - c.ne as i64 + c.nf as i64; }
    comps.sort_by_key(|c| std::cmp::Reverse(c.nf));
    let mut d = Diag { nv: m.v.len(), nf: m.f.len(), comps, oriented, star: None };
    if d.watertight() {
        let n = 8192;
        let mut hits = vec![];
        let ok = (0..n).filter(|&i| { bvh.ray([0.0; 3], fibonacci_dir(i, n), &mut hits); hits.len() == 1 }).count();
        d.star = Some(ok as f64 / n as f64);
    }
    d
}

impl Diag {
    pub fn watertight(&self) -> bool { !self.comps.is_empty() && self.comps.iter().all(|c| c.closed()) }
    pub fn boundary(&self) -> usize { self.comps.iter().map(|c| c.boundary).sum() }
    pub fn genus(&self) -> Option<i64> { if self.comps.len() == 1 { self.comps[0].genus() } else { None } }

    /// Ok(avertissement éventuel) si la méthode est applicable, Err(raison) sinon.
    pub fn avail(&self, m: Method) -> Result<Option<String>, String> {
        if !self.watertight() { return Err("surface ouverte ou non variété : réparer d'abord".into()); }
        match m {
            Method::C => Ok(None),
            _ if self.comps.len() > 1 => Err(format!("{} composantes : un seul morceau requis", self.comps.len())),
            Method::A => { let s = self.star.unwrap_or(0.0); Ok((s < 0.999).then(|| format!("{:.1} % de directions non étoilées", 100.0 * (1.0 - s)))) }
            Method::B => match self.genus() {
                _ if !self.oriented => Err("orientation incohérente".into()),
                Some(0) => Ok(None),
                g => Err(format!("genre {} : paramétrage sphérique impossible", g.unwrap_or(-1))),
            },
        }
    }

    /// Arbre de décision du tableau 4.
    pub fn recommend(&self) -> Option<Method> {
        if !self.watertight() { return None; }
        if self.comps.len() == 1 && self.star.unwrap_or(0.0) >= 0.999 { Some(Method::A) }
        else if self.avail(Method::B).is_ok() { Some(Method::B) } else { Some(Method::C) }
    }

    pub fn summary(&self) -> String {
        let mut s = format!("{} sommets, {} triangles, {} composante{}, {}", self.nv, self.nf, self.comps.len(), if self.comps.len() > 1 { "s" } else { "" },
            if self.watertight() { "étanche".to_string() } else { format!("non étanche ({} arêtes de bord)", self.boundary()) });
        if self.comps.len() == 1 { let c = &self.comps[0]; s += &format!(", χ = {}", c.euler); if let Some(g) = c.genus() { s += &format!(" (genre {g})"); } }
        if let Some(st) = self.star { s += &format!(", {:.1} % de rayons à sortie unique", 100.0 * st); }
        s
    }
}
