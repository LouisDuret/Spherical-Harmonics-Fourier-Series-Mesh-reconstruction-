//! Réparation : remaillage par voxels, iso-surface (marching tetrahedra), simplification.
use crate::{bvh::Bvh, mesh::*, pipeline::Ctl};
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

fn coord(x: usize, d: [usize; 3]) -> [usize; 3] { [x / (d[1] * d[2]), x / d[2] % d[1], x % d[2]] }

/// Dilatation morphologique (6-voisinage) répétée `r` fois.
fn dilate(s: &mut Vec<bool>, d: [usize; 3], r: usize) {
    let st = [d[1] * d[2], d[2], 1];
    for _ in 0..r {
        let o = s.clone();
        s.par_iter_mut().enumerate().for_each(|(x, v)| {
            if *v { return; }
            let c = coord(x, d);
            *v = (0..3).any(|k| (c[k] > 0 && o[x - st[k]]) || (c[k] + 1 < d[k] && o[x + st[k]]));
        });
    }
}

/// Remplissage par diffusion depuis `seed` dans les cases libres (lab = 0) où `pass` est vrai.
fn flood(lab: &mut [u32], l: u32, pass: impl Fn(usize) -> bool, seed: usize, d: [usize; 3]) -> usize {
    let st = [d[1] * d[2], d[2], 1];
    let (mut stack, mut n) = (vec![seed], 0);
    lab[seed] = l;
    while let Some(x) = stack.pop() {
        n += 1;
        let c = coord(x, d);
        for k in 0..3 {
            if c[k] > 0 && lab[x - st[k]] == 0 && pass(x - st[k]) { lab[x - st[k]] = l; stack.push(x - st[k]); }
            if c[k] + 1 < d[k] && lab[x + st[k]] == 0 && pass(x + st[k]) { lab[x + st[k]] = l; stack.push(x + st[k]); }
        }
    }
    n
}

/// Voxelisation, fermeture morphologique, remplissage, iso-surface lissée (pleine résolution).
pub fn voxel_remesh(m: &Mesh, n: usize, closing: usize, ctl: &Ctl) -> Result<Mesh, String> {
    ctl.step(0.05, "Voxelisation")?;
    let (lo, hi) = m.bbox();
    let ext = sub(hi, lo);
    let cell = ext.iter().cloned().fold(0.0, f64::max) / n.max(8) as f64;
    let pad = closing + 3;
    let d = [0, 1, 2].map(|k| (ext[k] / cell).ceil() as usize + 2 * pad + 1);
    let org = sub(lo, [pad as f64 * cell; 3]);
    let id = |g: [usize; 3]| (g[0] * d[1] + g[1]) * d[2] + g[2];
    let mut s = vec![false; d[0] * d[1] * d[2]];
    for t in 0..m.f.len() {
        let [a, b, c] = m.tri(t);
        let e = norm(sub(b, a)).max(norm(sub(c, a))).max(norm(sub(c, b)));
        let q = (e / (0.5 * cell)).ceil() as usize + 1;
        for i in 0..=q { for j in 0..=q - i {
            let p = add(a, add(mul(sub(b, a), i as f64 / q as f64), mul(sub(c, a), j as f64 / q as f64)));
            s[id([0, 1, 2].map(|k| ((p[k] - org[k]) / cell) as usize))] = true;
        } }
    }
    ctl.step(0.2, "Fermeture morphologique")?;
    dilate(&mut s, d, closing);
    let mut lab = vec![0u32; s.len()];
    flood(&mut lab, 1, |x| !s[x], 0, d);
    let mut outside: Vec<bool> = lab.iter().map(|&l| l == 1).collect();
    dilate(&mut outside, d, closing);
    // Trous plus larges que la fermeture : l'extérieur inonde l'intérieur. On rattrape ces cases par un
    // vote de parité des intersections le long des trois axes (intérieur si au moins deux axes l'affirment).
    let bvh = Bvh::new(m);
    let mut votes = vec![0u8; s.len()];
    for k in 0..3 {
        let (a, b) = ((k + 1) % 3, (k + 2) % 3);
        let cols: Vec<Vec<f64>> = (0..d[a] * d[b]).into_par_iter().map_init(Vec::new, |hits, c| {
            let (mut o, mut dir) = (org, [0.0; 3]);
            (o[a], o[b], dir[k]) = (o[a] + ((c / d[b]) as f64 + 0.5) * cell, o[b] + ((c % d[b]) as f64 + 0.5) * cell, 1.0);
            bvh.ray(o, dir, hits);
            hits.iter().map(|h| h.0).collect()
        }).collect();
        for (c, hits) in cols.iter().enumerate() {
            let (mut g, mut nh) = ([0; 3], 0);
            (g[a], g[b]) = (c / d[b], c % d[b]);
            for i in 0..d[k] {
                while nh < hits.len() && hits[nh] < (i as f64 + 0.5) * cell { nh += 1; }
                g[k] = i;
                votes[id(g)] += (nh % 2) as u8;
            }
        }
    }
    for x in 0..s.len() { outside[x] &= votes[x] < 2; }
    // Les poches que le vote laisse vides à l'intérieur ne doivent plus communiquer avec l'extérieur : on
    // bouche les canaux d'une ou deux cases (ouverture de l'extérieur), puis on comble ce qui est enclos.
    let mut solid: Vec<bool> = outside.iter().map(|o| !o).collect();
    dilate(&mut solid, d, 1);
    lab.fill(0);
    flood(&mut lab, 1, |x| !solid[x], 0, d);
    for x in 0..s.len() { outside[x] = lab[x] == 1; }
    dilate(&mut outside, d, 1);
    // Ne garder que les morceaux significatifs (≥ 1 % du plus gros).
    ctl.step(0.4, "Composantes")?;
    lab.fill(0);
    let mut sizes = vec![0usize];
    for x in 0..s.len() { if !outside[x] && lab[x] == 0 { let l = sizes.len() as u32; sizes.push(flood(&mut lab, l, |y| !outside[y], x, d)); } }
    let big = sizes.iter().cloned().max().unwrap_or(0);
    if big == 0 { return Err("réparation : volume vide (augmenter la résolution ou la fermeture)".into()); }
    ctl.step(0.55, "Lissage")?;
    let mut f: Vec<f32> = (0..s.len()).map(|x| if lab[x] > 0 && sizes[lab[x] as usize] * 100 >= big { -0.5 } else { 0.5 }).collect();
    let st = [d[1] * d[2], d[2], 1];
    for pass in 0..6 {
        let (k, o) = (pass % 3, f.clone());
        f.par_iter_mut().enumerate().for_each(|(x, v)| {
            let c = coord(x, d)[k];
            let a = if c > 0 { o[x - st[k]] } else { o[x] };
            let b = if c + 1 < d[k] { o[x + st[k]] } else { o[x] };
            *v = 0.25 * (a + b) + 0.5 * o[x];
        });
    }
    ctl.step(0.7, "Iso-surface")?;
    Ok(marching_tets(&f, d, add(org, [0.5 * cell; 3]), cell))
}

/// Iso-surface {f = 0} (intérieur : f < 0) par marching tetrahedra : toujours une variété orientée.
pub fn marching_tets(f: &[f32], d: [usize; 3], org: V3, cell: f64) -> Mesh {
    const P: [[usize; 3]; 6] = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    let total = (d[0] * d[1] * d[2]) as u64;
    let mut map: HashMap<u64, u32> = HashMap::new();
    let mut m = Mesh::default();
    let val = |x: usize| { let v = f[x] as f64; if v.abs() < 1e-9 { 1e-9 } else { v } };
    for i in 0..d[0] - 1 { for j in 0..d[1] - 1 { for k in 0..d[2] - 1 {
        let ix: [usize; 8] = std::array::from_fn(|c| ((i + (c & 1)) * d[1] + j + (c >> 1 & 1)) * d[2] + k + (c >> 2 & 1));
        let neg = ix.iter().filter(|&&x| val(x) < 0.0).count();
        if neg == 0 || neg == 8 { continue; }
        let pos = |c: usize| add(org, mul([(i + (c & 1)) as f64, (j + (c >> 1 & 1)) as f64, (k + (c >> 2 & 1)) as f64], cell));
        for p in P {
            let a = 1 << p[0];
            let tet = [0, a, a | 1 << p[1], 7];
            let (mut ins, mut outs) = (vec![], vec![]);
            for c in tet { if val(ix[c]) < 0.0 { ins.push(c) } else { outs.push(c) } }
            if ins.is_empty() || outs.is_empty() { continue; }
            let mut ev = |a: usize, b: usize| -> u32 {
                let (ga, gb) = (ix[a] as u64, ix[b] as u64);
                *map.entry(ga.min(gb) * total + ga.max(gb)).or_insert_with(|| {
                    let t = (val(ix[a]) / (val(ix[a]) - val(ix[b]))).clamp(0.02, 0.98);
                    m.v.push(add(mul(pos(a), 1.0 - t), mul(pos(b), t)));
                    m.v.len() as u32 - 1
                })
            };
            let tris: Vec<[u32; 3]> = match ins.len() {
                1 => vec![[ev(ins[0], outs[0]), ev(ins[0], outs[1]), ev(ins[0], outs[2])]],
                3 => vec![[ev(outs[0], ins[0]), ev(outs[0], ins[1]), ev(outs[0], ins[2])]],
                _ => { let q = [ev(ins[0], outs[0]), ev(ins[0], outs[1]), ev(ins[1], outs[1]), ev(ins[1], outs[0])]; vec![[q[0], q[1], q[2]], [q[0], q[2], q[3]]] }
            };
            let cen = |l: &[usize]| mul(l.iter().fold([0.0; 3], |s, &c| add(s, pos(c))), 1.0 / l.len() as f64);
            let g = sub(cen(&outs), cen(&ins));
            for mut t in tris {
                let (a, b, c) = (m.v[t[0] as usize], m.v[t[1] as usize], m.v[t[2] as usize]);
                if dot(cross(sub(b, a), sub(c, a)), g) < 0.0 { t.swap(1, 2); }
                m.f.push(t);
            }
        }
    } } }
    m
}

fn ring(vf: &[Vec<u32>], f: &[[u32; 3]], x: usize) -> Vec<u32> {
    let mut r: Vec<u32> = vf[x].iter().flat_map(|&t| f[t as usize]).filter(|&u| u != x as u32).collect();
    r.sort_unstable();
    r.dedup();
    r
}

/// Simplification par contraction des arêtes les plus courtes, topologie préservée (condition du lien).
pub fn simplify(m: &Mesh, target: usize) -> Mesh {
    let (mut v, mut f) = (m.v.clone(), m.f.clone());
    let mut vf: Vec<Vec<u32>> = vec![vec![]; v.len()];
    for (i, t) in f.iter().enumerate() { for &a in t { vf[a as usize].push(i as u32); } }
    let (mut dead, mut stamp, mut nf) = (vec![false; f.len()], vec![0u32; v.len()], f.len());
    let key = |v: &[V3], a: u32, b: u32| { let d = sub(v[a as usize], v[b as usize]); Reverse(dot(d, d).to_bits()) };
    let mut heap = BinaryHeap::new();
    for t in &f { for k in 0..3 { let (a, b) = (t[k], t[(k + 1) % 3]); if a < b { heap.push((key(&v, a, b), a, b, 0u32, 0u32)); } } }
    while nf > target.max(4) {
        let Some((_, a, b, sa, sb)) = heap.pop() else { break };
        let (ai, bi) = (a as usize, b as usize);
        if stamp[ai] != sa || stamp[bi] != sb || vf[ai].is_empty() || vf[bi].is_empty() { continue; }
        let shared: Vec<u32> = vf[ai].iter().copied().filter(|&t| f[t as usize].contains(&b)).collect();
        if shared.len() != 2 { continue; }
        let (na, nb) = (ring(&vf, &f, ai), ring(&vf, &f, bi));
        if na.iter().filter(|u| nb.binary_search(u).is_ok()).count() != 2 { continue; }
        let p = mul(add(v[ai], v[bi]), 0.5);
        let ok = vf[ai].iter().chain(vf[bi].iter()).filter(|t| !shared.contains(t)).all(|&t| {
            let t = f[t as usize];
            let o = t.map(|u| v[u as usize]);
            let q = t.map(|u| if u == a || u == b { p } else { v[u as usize] });
            let (n0, n1) = (cross(sub(o[1], o[0]), sub(o[2], o[0])), cross(sub(q[1], q[0]), sub(q[2], q[0])));
            dot(n0, n1) >= 0.2 * norm(n0) * norm(n1)
        });
        if !ok { continue; }
        v[ai] = p;
        for &t in &shared { dead[t as usize] = true; for u in f[t as usize] { vf[u as usize].retain(|&x| x != t); } }
        for t in std::mem::take(&mut vf[bi]) { for u in f[t as usize].iter_mut() { if *u == b { *u = a; } } vf[ai].push(t); }
        stamp[ai] += 1;
        stamp[bi] += 1;
        nf -= 2;
        for u in ring(&vf, &f, ai) { let (x, y) = (a.min(u), a.max(u)); heap.push((key(&v, x, y), x, y, stamp[x as usize], stamp[y as usize])); }
    }
    let mut out = Mesh { v, f: f.into_iter().zip(dead).filter(|x| !x.1).map(|x| x.0).collect() };
    out.weld();
    out
}
