//! Formule complète de l'objet sous forme d'arbres mathématiques (voir `tex`) : forme générale,
//! définitions, puis un nœud par terme du développement.
use crate::tex::*;
use harmomesh::pipeline::*;

pub type Rows = Vec<(Vec<i64>, Vec<f64>)>;

fn int(k: i64) -> String { k.to_string().replace('-', "−") }

/// |x| avec 6 décimales, ou mantisse × 10^n hors de la plage lisible.
pub fn numm(x: f64) -> M {
    let a = x.abs();
    if a != 0.0 && !(1e-4..1e6).contains(&a) { let e = a.log10().floor() as i32; sup(r(format!("{:.4} · 10", a / 10f64.powi(e))), r(int(e as i64))) } else { r(format!("{a:.6}")) }
}
fn sgn(x: f64, first: bool) -> M { r(if x < 0.0 { if first { "−" } else { " − " } } else if first { "" } else { " + " }) }
fn args() -> M { row![r("("), i("θ"), r(", "), i("φ"), r(")")] }
fn trig(f: &str, e: usize) -> Vec<M> { match e { 0 => vec![], 1 => vec![r(f), M::Sp(0.12), i("θ")], _ => vec![sup(r(f), r(e.to_string())), M::Sp(0.12), i("θ")] } }

/// Y(l, m) développée : polynôme en cos θ × sin^|m| θ × cos / sin(|m|φ), normalisation incluse.
fn ylm_dev(q: &[f64], m: i64) -> Vec<M> {
    let (am, k) = (m.unsigned_abs() as usize, if m == 0 { 1.0 } else { std::f64::consts::SQRT_2 });
    let monos: Vec<(usize, f64)> = q.iter().enumerate().rev().filter(|x| *x.1 != 0.0).map(|(e, c)| (e, c * k)).collect();
    let mono = |v: &mut Vec<M>, e: usize| if e > 0 { v.push(M::Sp(0.22)); v.extend(trig("cos", e)); };
    let mut v = vec![];
    if let [(e, c)] = monos[..] {
        v.push(if c < 0.0 { row![r("(−"), numm(c), r(")")] } else { numm(c) });
        mono(&mut v, e);
    } else {
        v.push(r("("));
        for (n, (e, c)) in monos.iter().enumerate() { v.extend([sgn(*c, n == 0), numm(*c)]); mono(&mut v, *e); }
        v.push(r(")"));
    }
    if am > 0 {
        v.push(M::Sp(0.22));
        v.extend(trig("sin", am));
        v.extend([M::Sp(0.22), r(if m > 0 { "cos" } else { "sin" }), M::Sp(0.12), r(if am > 1 { am.to_string() } else { String::new() }), i("φ")]);
    }
    v
}

/// Pour la méthode C : ne garder qu'un demi-espace de fréquences (f est réelle), constante en tête.
pub fn prepare(o: &Outcome, mut rows: Rows) -> Rows {
    if let Fit::Fourier(_) = o.fit {
        rows.retain(|(k, _)| k.iter().find(|x| **x != 0).is_none_or(|x| *x > 0));
        rows.sort_by_key(|(k, _)| k.iter().any(|x| *x != 0));
    }
    rows
}

pub struct Ctx<'a> { pub o: &'a Outcome, pub rows: &'a Rows, pub polys: &'a [Vec<f64>], pub explicit: bool }

impl Ctx<'_> {
    pub fn ncomp(&self) -> usize { match &self.o.fit { Fit::Sh(f) => f.nc, Fit::Fourier(_) => 1 } }

    /// Membre de gauche du développement de la composante `comp`.
    pub fn title(&self, comp: usize) -> M {
        match &self.o.fit {
            Fit::Sh(f) => row![i(if f.nc == 1 { "r" } else { ["x", "y", "z"][comp] }), args(), r(" =")],
            Fit::Fourier(_) => row![i("f"), r("("), i("X"), r(", "), i("Y"), r(", "), i("Z"), r(") =")],
        }
    }

    /// Le `t`-ième terme du développement de la composante `comp`.
    pub fn term(&self, comp: usize, t: usize) -> M {
        let (k, v) = &self.rows[t];
        if let Fit::Sh(_) = self.o.fit {
            let (l, m, c) = (k[0] as usize, k[1], v[comp]);
            let mut p = vec![sgn(c, t == 0), numm(c)];
            if self.explicit { p.push(r(" · ")); p.extend(ylm_dev(&self.polys[l * (l + 1) / 2 + m.unsigned_abs() as usize], m)); } else { p.extend([M::Sp(0.18), subsup(i("Y"), r(l.to_string()), r(int(m)))]); }
            return M::Row(p);
        }
        if k.iter().all(|x| *x == 0) { return row![sgn(v[0], true), numm(v[0])]; }
        let mut a = vec![r("2"), i("π"), r("(")];
        for (kk, name) in k.iter().zip(["u", "v", "w"]).filter(|x| *x.0 != 0) {
            let first = a.len() == 3;
            a.push(r(if *kk < 0 { if first { "−" } else { " − " } } else if first { "" } else { " + " }));
            if kk.abs() != 1 { a.push(r(kk.abs().to_string())); }
            a.push(i(name));
        }
        a.push(r(")"));
        let (arg, ca, cb) = (M::Row(a), 2.0 * v[0], -2.0 * v[1]);
        row![sgn(ca, false), numm(ca), M::Sp(0.15), r("cos"), M::Sp(0.12), arg.clone(), sgn(cb, false), numm(cb), M::Sp(0.15), r("sin"), M::Sp(0.12), arg]
    }

    /// Forme générale et définitions : (formule, remarque).
    pub fn header(&self, md: &Model, l: usize) -> Vec<(Option<M>, String)> {
        let (th, c) = (args(), md.xf.center);
        let f = |m: M| (Some(m), String::new());
        let frame = |top: &str| [
            f(row![i("C"), r(" = ("), sgn(c[0], true), numm(c[0]), r(", "), sgn(c[1], true), numm(c[1]), r(", "), sgn(c[2], true), numm(c[2]), r(")")]),
            f(row![i("s"), r(" = "), numm(md.xf.scale), r(",      "), i(top), r(format!(" = {l}"))]),
        ];
        let mut h = vec![];
        match &self.o.fit {
            Fit::Sh(fit) => {
                let dsum = row![sum(row![i("l"), r(" = 0")], i("L")), M::Sp(0.2), sum(row![i("m"), r(" = −"), i("l")], i("l")), M::Sp(0.25)];
                let y = |lo: M, hi: M| row![subsup(i("Y"), lo, hi), th.clone()];
                if fit.nc == 3 {
                    h.push(f(row![i("P"), th.clone(), r(" = "), i("C"), r(" + "), i("s"), r(" · ("), i("x"), th.clone(), r(", "), i("y"), th.clone(), r(", "), i("z"), th.clone(), r(")")]));
                    h.push((Some(row![i("x"), th.clone(), r(" = "), dsum, subsup(i("x"), i("l"), i("m")), M::Sp(0.15), y(i("l"), i("m"))]), "de même pour y et z ;  θ, φ : coordonnées du paramétrage sphérique".into()));
                } else {
                    h.push(f(row![i("P"), th.clone(), r(" = "), i("C"), r(" + "), i("s"), M::Sp(0.15), i("r"), th.clone(), r(" · (sin "), i("θ"), r(" cos "), i("φ"), r(", sin "), i("θ"), r(" sin "), i("φ"), r(", cos "), i("θ"), r(")")]));
                    h.push((Some(row![i("r"), th.clone(), r(" = "), dsum, subsup(i("r"), i("l"), i("m")), M::Sp(0.15), y(i("l"), i("m"))]), "θ, φ : direction depuis le centre C".into()));
                }
                h.extend(frame("L"));
                if self.explicit {
                    h.push((None, "Harmoniques développées en cos θ, sin θ, cos mφ, sin mφ, normalisation incluse. Forme exacte, mais mal conditionnée en virgule flottante au-delà de l ≈ 25 : pour calculer, préférer les récurrences (exports GLSL et Python).".into()));
                } else {
                    let (n, pl) = (subsup(i("N"), i("l"), i("m")), row![subsup(i("P"), i("l"), i("m")), r("(cos "), i("θ"), r(")")]);
                    h.push((Some(row![y(i("l"), i("m")), r(" = "), sqrt(r("2")), n.clone(), M::Sp(0.12), pl.clone(), r(" cos "), i("m"), i("φ"), r("        ("), i("m"), r(" > 0)")]), String::new()));
                    h.push(f(row![y(i("l"), r("0")), r(" = "), subsup(i("N"), i("l"), r("0")), M::Sp(0.12), subsup(i("P"), i("l"), r("0")), r("(cos "), i("θ"), r(")")]));
                    h.push(f(row![y(i("l"), row![r("−"), i("m")]), r(" = "), sqrt(r("2")), n.clone(), M::Sp(0.12), pl, r(" sin "), i("m"), i("φ"), r("        ("), i("m"), r(" > 0)")]));
                    h.push((Some(row![n, r(" = "), sqrt(row![frac(row![r("2"), i("l"), r(" + 1")], row![r("4"), i("π")]), r(" · "), frac(row![r("("), i("l"), r(" − "), i("m"), r(")!")], row![r("("), i("l"), r(" + "), i("m"), r(")!")])])]), "P : fonctions de Legendre associées, phase de Condon–Shortley".into()));
                }
            }
            Fit::Fourier(fit) => {
                let xyz = row![r("("), i("X"), r(", "), i("Y"), r(", "), i("Z"), r(")")];
                let ku = row![r("2"), i("π"), M::Sp(0.1), i("k"), r(" · "), i("u")];
                h.push((Some(row![i("f"), xyz.clone(), r(" = 0")]), "surface de l'objet ;  intérieur : f < 0 ;  s · f ≈ distance signée".into()));
                h.push(f(row![i("f"), r(" = "), sub(i("c"), r("0")), r(" + "), sum(row![r("|"), sub(i("k"), i("i")), r("| ≤ "), i("K")], r(" ")), M::Sp(0.2), r("["), sub(i("a"), i("k")), r(" cos("), ku.clone(), r(") + "), sub(i("b"), i("k")), r(" sin("), ku, r(")]")]));
                h.push(f(row![i("u"), r(" = ("), i("u"), r(", "), i("v"), r(", "), i("w"), r(") = "), frac(row![xyz, r(" − "), i("C")], row![r(format!("{} ", 2.0 * fit.h)), i("s")]), r(" + "), frac(r("1"), r("2"))]));
                h.extend(frame("K"));
            }
        }
        h
    }

    /// La formule complète en LaTeX (les longues sommes utilisent l'environnement dmath du paquet breqn).
    pub fn latex(&self, md: &Model, l: usize) -> String {
        let mut s = String::from("% HarmoMesh — formule complète ; nécessite \\usepackage{amsmath,breqn}\n");
        for (m, note) in self.header(md, l) {
            if let Some(m) = m { s += &format!("\\[ {} \\]", m.tex()); }
            if !note.is_empty() { s += &format!(" % {note}"); }
            s.push('\n');
        }
        for comp in 0..self.ncomp() {
            s += &format!("\\begin{{dmath*}}\n{}\n", self.title(comp).tex());
            for t in 0..self.rows.len() { s += &self.term(comp, t).tex(); s.push('\n'); }
            s += "\\end{dmath*}\n";
        }
        s
    }
}
