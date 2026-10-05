//! Mesh triangulaire : lecture/écriture (OBJ, PLY, STL, glTF), soudure, normalisation.
use std::collections::HashMap;
use std::path::Path;

pub type V3 = [f64; 3];
pub fn add(a: V3, b: V3) -> V3 { [a[0] + b[0], a[1] + b[1], a[2] + b[2]] }
pub fn sub(a: V3, b: V3) -> V3 { [a[0] - b[0], a[1] - b[1], a[2] - b[2]] }
pub fn mul(a: V3, s: f64) -> V3 { [a[0] * s, a[1] * s, a[2] * s] }
pub fn dot(a: V3, b: V3) -> f64 { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }
pub fn cross(a: V3, b: V3) -> V3 { [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]] }
pub fn norm(a: V3) -> f64 { dot(a, a).sqrt() }
pub fn unit(a: V3) -> V3 { let n = norm(a); if n > 0.0 { mul(a, 1.0 / n) } else { a } }

/// Transformation de normalisation : original = p * scale + center.
#[derive(Clone, Copy, Debug)]
pub struct Xf { pub center: V3, pub scale: f64 }
impl Xf {
    pub const ID: Xf = Xf { center: [0.0; 3], scale: 1.0 };
    pub fn apply(&self, p: V3) -> V3 { add(mul(p, self.scale), self.center) }
    /// Compose : `self` appliquée en premier (mesh d'origine), puis `inner`.
    pub fn then(&self, inner: Xf) -> Xf { Xf { center: self.apply(inner.center), scale: self.scale * inner.scale } }
}

#[derive(Clone, Default)]
pub struct Mesh { pub v: Vec<V3>, pub f: Vec<[u32; 3]> }

pub fn vertex_areas(v: &[V3], f: &[[u32; 3]]) -> Vec<f64> {
    let mut a = vec![0.0; v.len()];
    for t in f {
        let s = norm(cross(sub(v[t[1] as usize], v[t[0] as usize]), sub(v[t[2] as usize], v[t[0] as usize]))) / 6.0;
        for &i in t { a[i as usize] += s; }
    }
    a
}

impl Mesh {
    pub fn tri(&self, i: usize) -> [V3; 3] { let t = self.f[i]; [self.v[t[0] as usize], self.v[t[1] as usize], self.v[t[2] as usize]] }
    /// Normale non normalisée (norme = 2 × aire).
    pub fn fnormal(&self, i: usize) -> V3 { let [a, b, c] = self.tri(i); cross(sub(b, a), sub(c, a)) }
    pub fn area(&self) -> f64 { (0..self.f.len()).map(|i| norm(self.fnormal(i)) * 0.5).sum() }
    pub fn volume(&self) -> f64 { (0..self.f.len()).map(|i| { let [a, b, c] = self.tri(i); dot(a, cross(b, c)) / 6.0 }).sum() }
    pub fn bbox(&self) -> (V3, V3) {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for p in &self.v { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } }
        (lo, hi)
    }
    pub fn diagonal(&self) -> f64 { let (lo, hi) = self.bbox(); norm(sub(hi, lo)) }
    pub fn vertex_areas(&self) -> Vec<f64> { vertex_areas(&self.v, &self.f) }

    /// Centre de gravité du volume (ou de la surface si le volume n'est pas défini).
    pub fn centroid(&self) -> V3 {
        let (mut c, mut vol, mut ca, mut ar) = ([0.0; 3], 0.0, [0.0; 3], 0.0);
        for i in 0..self.f.len() {
            let [a, b, d] = self.tri(i);
            let s = add(add(a, b), d);
            let w = dot(a, cross(b, d)) / 6.0;
            c = add(c, mul(s, w / 4.0)); vol += w;
            let w = norm(self.fnormal(i)) * 0.5;
            ca = add(ca, mul(s, w / 3.0)); ar += w;
        }
        if vol.abs() > 1e-9 * self.diagonal().powi(3) { mul(c, 1.0 / vol) } else if ar > 0.0 { mul(ca, 1.0 / ar) } else { [0.0; 3] }
    }

    /// Centre à l'origine, mise à l'échelle dans la sphère unité.
    pub fn normalize(&mut self) -> Xf {
        let c = self.centroid();
        let s = self.v.iter().map(|&p| norm(sub(p, c))).fold(0.0, f64::max).max(1e-300);
        self.apply_inv(Xf { center: c, scale: s });
        Xf { center: c, scale: s }
    }
    pub fn apply_inv(&mut self, x: Xf) { for p in &mut self.v { *p = mul(sub(*p, x.center), 1.0 / x.scale); } }

    /// Fusionne les sommets dupliqués, retire les faces dégénérées et les sommets orphelins.
    pub fn weld(&mut self) {
        let eps = (self.diagonal() * 1e-9).max(1e-300);
        let mut map: HashMap<[i64; 3], u32> = HashMap::new();
        let mut v = vec![];
        let n = self.v.len() as u32;
        let re: Vec<u32> = self.v.iter().map(|p| *map.entry([0, 1, 2].map(|k| (p[k] / eps).round() as i64)).or_insert_with(|| { v.push(*p); v.len() as u32 - 1 })).collect();
        self.f = self.f.iter().filter(|t| t.iter().all(|&i| i < n)).map(|t| t.map(|i| re[i as usize])).filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]).collect();
        let mut used = vec![u32::MAX; v.len()];
        let mut out = vec![];
        for t in &mut self.f { for i in t.iter_mut() { if used[*i as usize] == u32::MAX { used[*i as usize] = out.len() as u32; out.push(v[*i as usize]); } *i = used[*i as usize]; } }
        self.v = out;
    }

    /// Étiquette de composante connexe par face (faces reliées par une arête), et nombre de composantes.
    pub fn components(&self) -> (Vec<u32>, usize) {
        let mut p: Vec<u32> = (0..self.f.len() as u32).collect();
        fn find(p: &mut [u32], mut x: u32) -> u32 { while p[x as usize] != x { p[x as usize] = p[p[x as usize] as usize]; x = p[x as usize]; } x }
        let mut edge: HashMap<(u32, u32), u32> = HashMap::new();
        for (i, t) in self.f.iter().enumerate() { for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            let j = *edge.entry((a.min(b), a.max(b))).or_insert(i as u32);
            let (x, y) = (find(&mut p, j), find(&mut p, i as u32));
            p[y as usize] = x;
        } }
        let mut ids: HashMap<u32, u32> = HashMap::new();
        let lab = (0..self.f.len() as u32).map(|i| { let r = find(&mut p, i); let n = ids.len() as u32; *ids.entry(r).or_insert(n) }).collect();
        (lab, ids.len())
    }

    pub fn vertex_normals(&self) -> Vec<[f32; 3]> {
        let mut n = vec![[0.0; 3]; self.v.len()];
        for (i, t) in self.f.iter().enumerate() { let fnm = self.fnormal(i); for &k in t { n[k as usize] = add(n[k as usize], fnm); } }
        n.into_iter().map(|x| unit(x).map(|c| c as f32)).collect()
    }

    pub fn load(path: &Path) -> Result<Mesh, String> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let rd = || std::fs::read(path).map_err(|e| format!("{} : {e}", path.display()));
        let mut m = match ext.as_str() {
            "obj" => load_obj(&String::from_utf8_lossy(&rd()?)),
            "ply" => load_ply(&rd()?)?,
            "stl" => load_stl(&rd()?),
            "gltf" | "glb" => load_gltf(path)?,
            _ => return Err(format!("format non pris en charge : .{ext} (OBJ, PLY, STL, glTF)")),
        };
        m.weld();
        if m.f.is_empty() { return Err("aucun triangle dans le fichier".into()); }
        Ok(m)
    }

    /// Écrit en OBJ ou PLY (binaire), dans le repère d'origine si `xf` est fourni.
    pub fn save(&self, path: &Path, xf: Option<Xf>) -> Result<(), String> {
        let x = xf.unwrap_or(Xf::ID);
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let mut o: Vec<u8> = vec![];
        if ext == "ply" {
            o.extend(format!("ply\nformat binary_little_endian 1.0\ncomment HarmoMesh\nelement vertex {}\nproperty float x\nproperty float y\nproperty float z\nelement face {}\nproperty list uchar int vertex_indices\nend_header\n", self.v.len(), self.f.len()).bytes());
            for p in &self.v { for c in x.apply(*p) { o.extend((c as f32).to_le_bytes()); } }
            for t in &self.f { o.push(3); for &i in t { o.extend((i as i32).to_le_bytes()); } }
        } else {
            let mut s = String::from("# HarmoMesh\n");
            for p in &self.v { let q = x.apply(*p); s += &format!("v {} {} {}\n", q[0], q[1], q[2]); }
            for t in &self.f { s += &format!("f {} {} {}\n", t[0] + 1, t[1] + 1, t[2] + 1); }
            o = s.into_bytes();
        }
        std::fs::write(path, o).map_err(|e| format!("{} : {e}", path.display()))
    }

    pub fn icosphere(sub: usize) -> Mesh {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let mut m = Mesh {
            v: [[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0], [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t], [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]].iter().map(|&p| unit(p)).collect(),
            f: vec![[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]],
        };
        for _ in 0..sub {
            let mut mid: HashMap<(u32, u32), u32> = HashMap::new();
            let mut f = vec![];
            for t in m.f.clone() {
                let h = [0, 1, 2].map(|k| { let (a, b) = (t[k], t[(k + 1) % 3]); *mid.entry((a.min(b), a.max(b))).or_insert_with(|| { m.v.push(unit(add(m.v[a as usize], m.v[b as usize]))); m.v.len() as u32 - 1 }) });
                f.extend([[t[0], h[0], h[2]], [t[1], h[1], h[0]], [t[2], h[2], h[1]], h]);
            }
            m.f = f;
        }
        m
    }

    pub fn torus(big: f64, small: f64, nu: usize, nv: usize) -> Mesh {
        let mut m = Mesh::default();
        for i in 0..nu { for j in 0..nv {
            let (u, w) = (i as f64 / nu as f64 * std::f64::consts::TAU, j as f64 / nv as f64 * std::f64::consts::TAU);
            m.v.push([(big + small * w.cos()) * u.cos(), (big + small * w.cos()) * u.sin(), small * w.sin()]);
            let id = |a: usize, b: usize| ((a % nu) * nv + b % nv) as u32;
            m.f.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
            m.f.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        } }
        m
    }
}

fn load_obj(s: &str) -> Mesh {
    let mut m = Mesh::default();
    for l in s.lines() {
        let mut t = l.split_whitespace();
        match t.next() {
            Some("v") => { let p: Vec<f64> = t.take(3).filter_map(|x| x.parse().ok()).collect(); if p.len() == 3 { m.v.push([p[0], p[1], p[2]]); } }
            Some("f") => {
                let n = m.v.len() as i64;
                let idx: Vec<u32> = t.filter_map(|x| x.split('/').next()?.parse::<i64>().ok()).map(|i| (if i < 0 { n + i } else { i - 1 }) as u32).collect();
                for i in 1..idx.len().saturating_sub(1) { m.f.push([idx[0], idx[i], idx[i + 1]]); }
            }
            _ => {}
        }
    }
    m
}

fn load_stl(d: &[u8]) -> Mesh {
    let mut m = Mesh::default();
    let n = if d.len() >= 84 { u32::from_le_bytes([d[80], d[81], d[82], d[83]]) as usize } else { 0 };
    if d.len() >= 84 && d.len() == 84 + 50 * n {
        for i in 0..n { for k in 0..3 {
            let o = 84 + 50 * i + 12 + 12 * k;
            m.v.push([0, 1, 2].map(|c| f32::from_le_bytes(d[o + 4 * c..o + 4 * c + 4].try_into().unwrap()) as f64));
        } }
    } else {
        let s = String::from_utf8_lossy(d);
        let mut t = s.split_whitespace();
        while let Some(w) = t.next() { if w == "vertex" { let p: Vec<f64> = t.by_ref().take(3).filter_map(|x| x.parse().ok()).collect(); if p.len() == 3 { m.v.push([p[0], p[1], p[2]]); } } }
    }
    m.f = (0..m.v.len() as u32 / 3).map(|i| [3 * i, 3 * i + 1, 3 * i + 2]).collect();
    m
}

fn load_ply(d: &[u8]) -> Result<Mesh, String> {
    let key = b"end_header";
    let hp = d.windows(key.len()).position(|w| w == key).ok_or("PLY : en-tête invalide")?;
    let mut pos = hp + key.len();
    while pos < d.len() && d[pos] != b'\n' { pos += 1; }
    pos += 1;
    let head = String::from_utf8_lossy(&d[..hp]).to_string();
    let (mut ascii, mut big) = (false, false);
    let mut els: Vec<(String, usize, Vec<(String, Option<String>, String)>)> = vec![];
    for l in head.lines() {
        let t: Vec<&str> = l.split_whitespace().collect();
        match t.as_slice() {
            ["format", f, ..] => { ascii = *f == "ascii"; big = *f == "binary_big_endian"; }
            ["element", n, c] => els.push((n.to_string(), c.parse().map_err(|_| "PLY : élément invalide")?, vec![])),
            ["property", "list", ct, it, n] => if let Some(e) = els.last_mut() { e.2.push((n.to_string(), Some(ct.to_string()), it.to_string())); },
            ["property", ty, n] => if let Some(e) = els.last_mut() { e.2.push((n.to_string(), None, ty.to_string())); },
            _ => {}
        }
    }
    let text = if ascii { String::from_utf8_lossy(&d[pos.min(d.len())..]).to_string() } else { String::new() };
    let mut toks = text.split_ascii_whitespace();
    let mut rd = |ty: &str| -> Result<f64, String> {
        if ascii { return toks.next().and_then(|t| t.parse().ok()).ok_or("PLY tronqué".to_string()); }
        let n = match ty { "char" | "uchar" | "int8" | "uint8" => 1, "short" | "ushort" | "int16" | "uint16" => 2, "double" | "float64" => 8, _ => 4 };
        let b = d.get(pos..pos + n).ok_or("PLY tronqué")?;
        pos += n;
        let mut a = [0u8; 8];
        a[..n].copy_from_slice(b);
        if big { a[..n].reverse(); }
        Ok(match ty {
            "char" | "int8" => a[0] as i8 as f64, "uchar" | "uint8" => a[0] as f64,
            "short" | "int16" => i16::from_le_bytes([a[0], a[1]]) as f64, "ushort" | "uint16" => u16::from_le_bytes([a[0], a[1]]) as f64,
            "int" | "int32" => i32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64, "uint" | "uint32" => u32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64,
            "double" | "float64" => f64::from_le_bytes(a), _ => f32::from_le_bytes([a[0], a[1], a[2], a[3]]) as f64,
        })
    };
    let mut m = Mesh::default();
    for (name, cnt, props) in &els { for _ in 0..*cnt {
        let mut p = [0.0; 3];
        for (pn, list, ty) in props {
            if let Some(ct) = list {
                let n = rd(ct)? as usize;
                let idx = (0..n).map(|_| rd(ty).map(|x| x as u32)).collect::<Result<Vec<u32>, String>>()?;
                if name == "face" && pn.starts_with("vertex_ind") { for i in 1..n.saturating_sub(1) { m.f.push([idx[0], idx[i], idx[i + 1]]); } }
            } else {
                let x = rd(ty)?;
                match pn.as_str() { "x" => p[0] = x, "y" => p[1] = x, "z" => p[2] = x, _ => {} }
            }
        }
        if name == "vertex" { m.v.push(p); }
    } }
    Ok(m)
}

fn load_gltf(path: &Path) -> Result<Mesh, String> {
    let (doc, bufs, _) = gltf::import(path).map_err(|e| format!("glTF : {e}"))?;
    let mut m = Mesh::default();
    type M4 = [[f32; 4]; 4];
    fn node(n: &gltf::Node, parent: M4, bufs: &[gltf::buffer::Data], m: &mut Mesh) {
        let l = n.transform().matrix();
        let mut t = [[0f32; 4]; 4];
        for c in 0..4 { for r in 0..4 { t[c][r] = (0..4).map(|k| parent[k][r] * l[c][k]).sum(); } }
        if let Some(mesh) = n.mesh() { for p in mesh.primitives() {
            if p.mode() != gltf::mesh::Mode::Triangles { continue; }
            let r = p.reader(|b| Some(&bufs[b.index()]));
            let Some(pos) = r.read_positions() else { continue };
            let base = m.v.len() as u32;
            for q in pos { m.v.push([0, 1, 2].map(|r| (t[0][r] * q[0] + t[1][r] * q[1] + t[2][r] * q[2] + t[3][r]) as f64)); }
            let cnt = m.v.len() as u32 - base;
            let idx: Vec<u32> = match r.read_indices() { Some(i) => i.into_u32().collect(), None => (0..cnt).collect() };
            for c in idx.chunks_exact(3) { m.f.push([base + c[0], base + c[1], base + c[2]]); }
        } }
        for c in n.children() { node(&c, t, bufs, m); }
    }
    let id: M4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
    for s in doc.scenes() { for n in s.nodes() { node(&n, id, &bufs, &mut m); } }
    Ok(m)
}
