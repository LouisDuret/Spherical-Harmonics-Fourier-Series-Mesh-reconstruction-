//! BVH : lancer de rayons et distance au mesh.
use crate::mesh::*;

struct Node { lo: V3, hi: V3, a: u32, n: u32, leaf: bool }
pub struct Bvh { nodes: Vec<Node>, tri: Vec<[V3; 3]>, id: Vec<u32> }

impl Bvh {
    pub fn new(m: &Mesh) -> Bvh {
        let mut it: Vec<(V3, u32)> = (0..m.f.len()).map(|i| { let t = m.tri(i); (mul(add(add(t[0], t[1]), t[2]), 1.0 / 3.0), i as u32) }).collect();
        let mut b = Bvh { nodes: Vec::with_capacity(m.f.len() / 2 + 1), tri: vec![], id: vec![] };
        b.build(m, &mut it, 0);
        b.tri = it.iter().map(|x| m.tri(x.1 as usize)).collect();
        b.id = it.iter().map(|x| x.1).collect();
        b
    }

    fn build(&mut self, m: &Mesh, it: &mut [(V3, u32)], off: usize) -> usize {
        let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
        for x in it.iter() { for p in m.tri(x.1 as usize) { for k in 0..3 { lo[k] = lo[k].min(p[k]); hi[k] = hi[k].max(p[k]); } } }
        let me = self.nodes.len();
        self.nodes.push(Node { lo, hi, a: off as u32, n: it.len() as u32, leaf: true });
        if it.len() > 4 {
            let ax = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b]))).unwrap();
            let mid = it.len() / 2;
            it.select_nth_unstable_by(mid, |a, b| a.0[ax].total_cmp(&b.0[ax]));
            let (l, r) = it.split_at_mut(mid);
            self.build(m, l, off);
            let right = self.build(m, r, off + mid);
            self.nodes[me].a = right as u32;
            self.nodes[me].leaf = false;
        }
        me
    }

    /// Distance² au mesh (bornée par `best`) et face la plus proche (u32::MAX si aucune sous la borne).
    pub fn closest(&self, p: V3, mut best: f64) -> (f64, u32) {
        let box_d2 = |nd: &Node| (0..3).map(|k| { let d = (nd.lo[k] - p[k]).max(p[k] - nd.hi[k]).max(0.0); d * d }).sum::<f64>();
        let (mut face, mut st, mut sp) = (u32::MAX, [0u32; 96], 1);
        while sp > 0 {
            sp -= 1;
            let i = st[sp];
            let nd = &self.nodes[i as usize];
            if box_d2(nd) >= best { continue; }
            if nd.leaf {
                for j in nd.a as usize..(nd.a + nd.n) as usize { let d = tri_d2(p, &self.tri[j]); if d < best { best = d; face = self.id[j]; } }
            } else {
                let (l, r) = (i + 1, nd.a);
                if box_d2(&self.nodes[l as usize]) < box_d2(&self.nodes[r as usize]) { st[sp] = r; st[sp + 1] = l; } else { st[sp] = l; st[sp + 1] = r; }
                sp += 2;
            }
        }
        (best, face)
    }

    /// Toutes les intersections (t, face) du rayon o + t·d, triées, doublons d'arête fusionnés.
    pub fn ray(&self, o: V3, d: V3, hits: &mut Vec<(f64, u32)>) {
        hits.clear();
        if self.tri.is_empty() { return; }
        let inv = [1.0 / d[0], 1.0 / d[1], 1.0 / d[2]];
        let (mut st, mut sp) = ([0u32; 96], 1);
        while sp > 0 {
            sp -= 1;
            let i = st[sp];
            let nd = &self.nodes[i as usize];
            let (mut t0, mut t1) = (0.0f64, f64::MAX);
            for k in 0..3 { let (x, y) = ((nd.lo[k] - o[k]) * inv[k], (nd.hi[k] - o[k]) * inv[k]); t0 = t0.max(x.min(y)); t1 = t1.min(x.max(y)); }
            if t0 > t1 { continue; }
            if nd.leaf {
                for j in nd.a as usize..(nd.a + nd.n) as usize { if let Some(t) = ray_tri(o, d, &self.tri[j]) { hits.push((t, self.id[j])); } }
            } else { st[sp] = i + 1; st[sp + 1] = nd.a; sp += 2; }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        hits.dedup_by(|b, a| (b.0 - a.0).abs() < 1e-9 * (1.0 + a.0));
    }
}

fn ray_tri(o: V3, d: V3, t: &[V3; 3]) -> Option<f64> {
    let (e1, e2) = (sub(t[1], t[0]), sub(t[2], t[0]));
    let h = cross(d, e2);
    let a = dot(e1, h);
    if a.abs() < 1e-300 { return None; }
    let (f, s) = (1.0 / a, sub(o, t[0]));
    let u = f * dot(s, h);
    if !(0.0..=1.0).contains(&u) { return None; }
    let q = cross(s, e1);
    let v = f * dot(d, q);
    if v < 0.0 || u + v > 1.0 { return None; }
    let tt = f * dot(e2, q);
    (tt > 1e-12).then_some(tt)
}

/// Distance² d'un point à un triangle (Ericson).
fn tri_d2(p: V3, t: &[V3; 3]) -> f64 {
    let (a, b, c) = (t[0], t[1], t[2]);
    let (ab, ac, ap) = (sub(b, a), sub(c, a), sub(p, a));
    let (d1, d2) = (dot(ab, ap), dot(ac, ap));
    if d1 <= 0.0 && d2 <= 0.0 { return dot(ap, ap); }
    let bp = sub(p, b);
    let (d3, d4) = (dot(ab, bp), dot(ac, bp));
    if d3 >= 0.0 && d4 <= d3 { return dot(bp, bp); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { let q = sub(ap, mul(ab, d1 / (d1 - d3))); return dot(q, q); }
    let cp = sub(p, c);
    let (d5, d6) = (dot(ab, cp), dot(ac, cp));
    if d6 >= 0.0 && d5 <= d6 { return dot(cp, cp); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { let q = sub(ap, mul(ac, d2 / (d2 - d6))); return dot(q, q); }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 { let q = sub(bp, mul(sub(c, b), (d4 - d3) / ((d4 - d3) + (d5 - d6)))); return dot(q, q); }
    let den = 1.0 / (va + vb + vc);
    let q = sub(ap, add(mul(ab, vb * den), mul(ac, vc * den)));
    dot(q, q)
}
