//! Petit moteur de composition mathématique pour egui : italiques, indices, exposants,
//! fractions, racines et sommes avec bornes. Sait aussi produire le LaTeX équivalent.
use eframe::egui::{self, vec2, Color32, FontFamily, FontId, Pos2, Stroke, Vec2};
use std::sync::Arc;

#[derive(Clone)]
pub enum M {
    /// Texte droit (nombres, opérateurs, noms de fonctions).
    R(String),
    /// Texte italique (variables).
    I(String),
    Row(Vec<M>),
    Sup(Box<M>, Box<M>),
    Sub(Box<M>, Box<M>),
    SubSup(Box<M>, Box<M>, Box<M>),
    Frac(Box<M>, Box<M>),
    Sqrt(Box<M>),
    /// Somme avec borne inférieure et supérieure.
    Sum(Box<M>, Box<M>),
    /// Espace, en em.
    Sp(f32),
}

macro_rules! row { ($($x:expr),* $(,)?) => { $crate::tex::M::Row(vec![$($x),*]) } }

pub fn r(s: impl Into<String>) -> M { M::R(s.into()) }
pub fn i(s: impl Into<String>) -> M { M::I(s.into()) }
pub fn sup(a: M, b: M) -> M { M::Sup(a.into(), b.into()) }
pub fn sub(a: M, b: M) -> M { M::Sub(a.into(), b.into()) }
pub fn subsup(a: M, lo: M, hi: M) -> M { M::SubSup(a.into(), lo.into(), hi.into()) }
pub fn frac(a: M, b: M) -> M { M::Frac(a.into(), b.into()) }
pub fn sqrt(a: M) -> M { M::Sqrt(a.into()) }
pub fn sum(lo: M, hi: M) -> M { M::Sum(lo.into(), hi.into()) }

/// Charge des polices mathématiques du système (Cambria, sinon Times ou DejaVu) dans les familles
/// « math » et « mathit » ; à défaut, les polices d'egui servent de repli.
pub fn install_fonts(ctx: &egui::Context) {
    let mut f = egui::FontDefinitions::default();
    let base = f.families[&FontFamily::Proportional].clone();
    let mut load = |name: &str, files: &[(&str, u32)]| -> Option<String> {
        files.iter().find_map(|(p, idx)| std::fs::read(p).ok().map(|b| {
            let mut d = egui::FontData::from_owned(b);
            d.index = *idx;
            f.font_data.insert(name.into(), Arc::new(d));
            name.to_string()
        }))
    };
    let up = load("math-roman", &[("C:/Windows/Fonts/cambria.ttc", 0), ("C:/Windows/Fonts/times.ttf", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 0), ("/System/Library/Fonts/Supplemental/Times New Roman.ttf", 0)]);
    let sym = load("math-symbols", &[("C:/Windows/Fonts/cambria.ttc", 1)]);
    let it = load("math-italic", &[("C:/Windows/Fonts/cambriai.ttf", 0), ("C:/Windows/Fonts/timesi.ttf", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf", 0), ("/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf", 0)]);
    let chain = |first: Option<String>| first.into_iter().chain(up.clone()).chain(sym.clone()).chain(base.clone()).collect::<Vec<_>>();
    f.families.insert(FontFamily::Name("math".into()), chain(None));
    f.families.insert(FontFamily::Name("mathit".into()), chain(it));
    ctx.set_fonts(f);
}

enum Cmd { Text(Vec2, Arc<egui::Galley>), Line(Vec2, Vec2, f32) }

/// Formule mise en page : largeur, hauteur au-dessus et au-dessous de la ligne de base.
pub struct Laid { pub w: f32, pub up: f32, pub down: f32, cmds: Vec<Cmd> }

impl Laid {
    fn empty() -> Laid { Laid { w: 0.0, up: 0.0, down: 0.0, cmds: vec![] } }
    /// Ajoute `o` décalé de (dx, dy), dy positif vers le bas.
    fn push(&mut self, o: Laid, dx: f32, dy: f32) {
        let d = vec2(dx, dy);
        self.cmds.extend(o.cmds.into_iter().map(|c| match c { Cmd::Text(p, g) => Cmd::Text(p + d, g), Cmd::Line(a, b, t) => Cmd::Line(a + d, b + d, t) }));
        (self.w, self.up, self.down) = (self.w.max(dx + o.w), self.up.max(o.up - dy), self.down.max(o.down + dy));
    }
    /// Dessine la formule, `origin` étant l'extrémité gauche de la ligne de base.
    pub fn paint(&self, p: &egui::Painter, origin: Pos2, col: Color32) {
        for c in &self.cmds { match c {
            Cmd::Text(o, g) => p.galley(origin + *o, g.clone(), col),
            Cmd::Line(a, b, t) => { p.line_segment([origin + *a, origin + *b], Stroke::new(*t, col)); }
        } }
    }
}

/// Met en page la formule `m` au corps `s` (pixels).
pub fn lay(p: &egui::Painter, m: &M, s: f32, col: Color32) -> Laid {
    let text = |t: &str, fam: &str, s: f32| -> Laid {
        let g = p.layout_no_wrap(t.to_owned(), FontId::new(s, FontFamily::Name(fam.into())), col);
        let h = g.size().y;
        let asc = g.rows.first().and_then(|r| r.glyphs.first()).map_or(0.78 * h, |g| g.font_ascent);
        Laid { w: g.size().x, up: asc, down: h - asc, cmds: vec![Cmd::Text(vec2(0.0, -asc), g)] }
    };
    let (axis, thick, small) = (-0.27 * s, (s * 0.06).max(1.0), 0.7 * s);
    match m {
        M::R(t) => text(t, "math", s),
        M::I(t) => text(t, "mathit", s),
        M::Sp(e) => Laid { w: e * s, ..Laid::empty() },
        M::Row(v) => v.iter().fold(Laid::empty(), |mut o, c| { let x = o.w; o.push(lay(p, c, s, col), x, 0.0); o }),
        M::Sup(b, e) => { let mut o = lay(p, b, s, col); let x = o.w; o.push(lay(p, e, small, col), x + 0.04 * s, -0.36 * s); o }
        M::Sub(b, e) => { let mut o = lay(p, b, s, col); let x = o.w; o.push(lay(p, e, small, col), x + 0.02 * s, 0.2 * s); o }
        M::SubSup(b, lo, hi) => {
            let mut o = lay(p, b, s, col);
            let x = o.w;
            o.push(lay(p, hi, small, col), x + 0.04 * s, -0.36 * s);
            o.push(lay(p, lo, small, col), x + 0.02 * s, 0.2 * s);
            o
        }
        M::Frac(n, d) => {
            let (n, d) = (lay(p, n, 0.92 * s, col), lay(p, d, 0.92 * s, col));
            let w = n.w.max(d.w) + 0.3 * s;
            let (nx, dx, ny, dy) = ((w - n.w) / 2.0, (w - d.w) / 2.0, axis - 0.02 * s - n.down, axis + d.up * 0.9);
            let mut o = Laid::empty();
            o.push(n, nx, ny);
            o.push(d, dx, dy);
            o.cmds.push(Cmd::Line(vec2(0.04 * s, axis), vec2(w - 0.04 * s, axis), thick));
            o.w = w;
            o
        }
        M::Sqrt(x) => {
            let c = lay(p, x, s, col);
            let (top, bot, x0, cw) = (-(c.up * 0.84 + 0.1 * s), c.down * 0.6, 0.5 * s, c.w);
            let mut o = Laid::empty();
            o.push(c, x0, 0.0);
            let pts = [vec2(0.02 * s, top + (bot - top) * 0.62), vec2(0.14 * s, top + (bot - top) * 0.55), vec2(0.28 * s, bot), vec2(0.46 * s, top), vec2(x0 + cw + 0.08 * s, top)];
            o.cmds.extend(pts.windows(2).map(|q| Cmd::Line(q[0], q[1], thick)));
            (o.w, o.up) = (x0 + cw + 0.16 * s, o.up.max(-top + thick));
            o
        }
        M::Sum(lo, hi) => {
            let big = 1.7 * s;
            let (sg, lo, hi) = (text("∑", "math", big), lay(p, lo, 0.62 * s, col), lay(p, hi, 0.62 * s, col));
            let w = sg.w.max(lo.w).max(hi.w);
            let (dy, sx, lx, hx, lu, hd) = (0.16 * s, (w - sg.w) / 2.0, (w - lo.w) / 2.0, (w - hi.w) / 2.0, lo.up, hi.down);
            let mut o = Laid::empty();
            o.push(sg, sx, dy);
            o.push(lo, lx, dy + 0.2 * big + lu);
            o.push(hi, hx, dy - 0.74 * big - hd);
            o.w = w + 0.08 * s;
            o
        }
    }
}

impl M {
    /// Code LaTeX équivalent.
    pub fn tex(&self) -> String {
        match self {
            M::R(t) | M::I(t) => [("−", "-"), ("θ", "\\theta "), ("φ", "\\varphi "), ("π", "\\pi "), ("·", "\\cdot "), ("≤", "\\le "), ("sin", "\\sin "), ("cos", "\\cos ")].iter().fold(t.clone(), |s, (a, b)| s.replace(a, b)),
            M::Row(v) => v.iter().map(|m| m.tex()).collect(),
            M::Sup(a, b) => format!("{}^{{{}}}", a.tex(), b.tex()),
            M::Sub(a, b) => format!("{}_{{{}}}", a.tex(), b.tex()),
            M::SubSup(a, lo, hi) => format!("{}_{{{}}}^{{{}}}", a.tex(), lo.tex(), hi.tex()),
            M::Frac(a, b) => format!("\\frac{{{}}}{{{}}}", a.tex(), b.tex()),
            M::Sqrt(a) => format!("\\sqrt{{{}}}", a.tex()),
            M::Sum(lo, hi) => format!("\\sum_{{{}}}^{{{}}}", lo.tex(), hi.tex()),
            M::Sp(_) => "\\,".into(),
        }
    }
}
