//! Export : coefficients (JSON, CSV), formule exécutable (GLSL, Python), mesh reconstruit.
use crate::{pipeline::*, sh::{Sh, Window}};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Coefficients tronqués au degré `l`, fenêtre appliquée : lignes [l, m, valeurs…] ou [kx, ky, kz, re, im].
pub fn rows(o: &Outcome, l: usize, win: Window) -> Vec<(Vec<i64>, Vec<f64>)> {
    match &o.fit {
        Fit::Sh(f) => (0..(l + 1) * (l + 1)).map(|i| {
            let d = (i as f64).sqrt().floor() as i64;
            (vec![d, i as i64 - d * d - d], f.coeffs[i * f.nc..(i + 1) * f.nc].iter().map(|c| c * win.sigma(d as f64, l)).collect())
        }).collect(),
        Fit::Fourier(f) => (0..f.c.len()).filter_map(|i| { let g = f.sigma(i, l, win); (g != 0.0).then(|| (f.k_of(i).to_vec(), vec![f.c[i].re * g, f.c[i].im * g])) }).collect(),
    }
}

pub fn header(o: &Outcome) -> Vec<&'static str> {
    match &o.fit { Fit::Sh(f) if f.nc == 1 => vec!["l", "m", "r"], Fit::Sh(_) => vec!["l", "m", "x", "y", "z"], Fit::Fourier(_) => vec!["kx", "ky", "kz", "re", "im"] }
}

pub fn csv(o: &Outcome, l: usize, win: Window, sep: &str) -> String {
    let mut s = header(o).join(sep) + "\n";
    for (k, v) in rows(o, l, win) { s += &k.iter().map(|x| x.to_string()).chain(v.iter().map(|x| format!("{x:e}"))).collect::<Vec<_>>().join(sep); s.push('\n'); }
    s
}

pub fn json(md: &Model, o: &Outcome, p: &Params, l: usize, win: Window) -> Value {
    let r = rows(o, l, win);
    let coef: Vec<Value> = r.iter().map(|(_, v)| if v.len() == 1 { json!(v[0]) } else { json!(v) }).collect();
    let mut j = json!({
        "software": format!("HarmoMesh {}", env!("CARGO_PKG_VERSION")),
        "source": md.name, "method": o.method.letter(), "lmax": l,
        "convention": {
            "basis": "harmoniques sphériques réelles orthonormées, phase de Condon-Shortley",
            "theta": "colatitude depuis +z", "phi": "longitude atan2(y, x)", "index": "l*l + l + m",
            "m>0": "sqrt(2) N P_l^m(cos θ) cos(mφ)", "m<0": "sqrt(2) N P_l^|m|(cos θ) sin(|m|φ)",
        },
        "normalization": { "center": md.xf.center, "scale": md.xf.scale, "rotation": "identité", "formula": "original = p * scale + center" },
        "params": { "lambda": p.lambda, "weight": p.weight.name(), "window": win.name(), "area_fix": p.area_fix, "cmcf_iters": p.cmcf_iters, "cmcf_delta": p.cmcf_delta, "grid": p.grid, "repaired": md.repaired },
        "metrics": { "chamfer_pct": o.metrics.chamfer, "hausdorff_pct": o.metrics.hausdorff, "at_lmax": o.lmax },
        "coefficients": coef,
    });
    if let Some(pi) = &o.param { j["param_alignment"] = json!(pi.rot); }
    if let Fit::Fourier(f) = &o.fit {
        j["kmax"] = json!(l); j["half_size"] = json!(f.h); j["grid"] = json!(f.m);
        j["convention"] = json!({ "formula": "f(x) = Re Σ_k c_k exp(2πi k·u), u = (p + half_size) / (2 half_size), p normalisé", "order": "kx, ky, kz de -kmax à kmax (kz varie le plus vite)", "surface": "f = 0, intérieur f < 0" });
    }
    j
}

fn num(x: f64) -> String { let a = x.abs(); if a != 0.0 && !(1e-4..1e6).contains(&a) { format!("{x:+.5e}") } else { format!("{x:+.6}") } }

/// Y[l,m] sous forme développée : polynôme en cos θ × sin^|m| θ × cos / sin(|m|φ), normalisation incluse.
fn ylm_explicit(q: &[f64], m: i64) -> String {
    let (am, k) = (m.unsigned_abs() as usize, if m == 0 { 1.0 } else { std::f64::consts::SQRT_2 });
    let pw = |b: &str, e: usize| match e { 0 => String::new(), 1 => format!("{b} θ"), _ => format!("{b}^{e} θ") };
    let monos: Vec<String> = q.iter().enumerate().rev().filter(|(_, c)| **c != 0.0).map(|(e, c)| if e == 0 { num(c * k) } else { format!("{}·{}", num(c * k), pw("cos", e)) }).collect();
    let mut parts = vec![if monos.len() == 1 { monos[0].clone() } else { format!("({})", monos.join(" ")) }];
    if am > 0 { parts.push(pw("sin", am)); parts.push(format!("{} {}φ", if m > 0 { "cos" } else { "sin" }, if am == 1 { String::new() } else { am.to_string() })); }
    if monos.len() == 1 { format!("({})", parts.join("·")) } else { parts.join("·") }
}

/// Formule complète de l'objet au degré `l`, une ligne par terme, dans le repère d'origine.
/// `explicit` développe chaque harmonique en cos θ, sin θ, cos mφ, sin mφ (méthodes A et B).
pub fn formula(md: &Model, o: &Outcome, l: usize, win: Window, explicit: bool) -> Vec<String> {
    let r = rows(o, l, win);
    let (c, s) = (md.xf.center, md.xf.scale);
    let frame = format!("centre = ({}, {}, {})          échelle = {}", num(c[0]), num(c[1]), num(c[2]), num(s));
    let mut t = vec![format!("HarmoMesh — formule complète de « {} »", md.name)];
    match &o.fit {
        Fit::Fourier(f) => {
            let half: Vec<_> = r.iter().filter(|(k, _)| k.iter().find(|x| **x != 0).is_some_and(|x| *x > 0)).collect();
            let c0 = r.iter().find(|(k, _)| k.iter().all(|x| *x == 0)).map_or(0.0, |x| x.1[0]);
            t.push(format!("Méthode C · série de Fourier 3D de la distance signée, K = {l} : {} termes ({} coefficients réels)", half.len() + 1, r.len()));
            t.extend([String::new(), "Surface de l'objet : f(X, Y, Z) = 0          (intérieur : f < 0 ; f × échelle ≈ distance signée)".into(),
                "f(X, Y, Z) = c0 + Σ [ a·cos 2π(kx·u + ky·v + kz·w) + b·sin 2π(kx·u + ky·v + kz·w) ]".into(),
                format!("(u, v, w) = ((X, Y, Z) - centre) / ({} · échelle) + 1/2", 2.0 * f.h), frame, String::new(), "f(X, Y, Z) =".into(), format!("  {}", num(c0))]);
            t.extend(half.iter().map(|(k, v)| { let q = format!("2π({:+}u {:+}v {:+}w)", k[0], k[1], k[2]); format!("  {}·cos {q} {}·sin {q}", num(2.0 * v[0]), num(-2.0 * v[1])) }));
        }
        Fit::Sh(f) => {
            let b = f.nc == 3;
            t.push(format!("Méthode {}, degré L = {l} : {} termes ({} coefficients réels)", o.method.name(), r.len(), r.len() * f.nc));
            t.push(String::new());
            if b {
                t.push("P(θ, φ) = centre + échelle · Σ c[l,m] · Y[l,m](θ, φ)          avec c[l,m] = (x, y, z)".into());
                t.push("θ ∈ [0, π], φ ∈ [-π, π] : coordonnées du paramétrage sphérique (colatitude depuis +z, longitude)".into());
            } else {
                t.push("P(θ, φ) = centre + échelle · r(θ, φ) · (sin θ cos φ, sin θ sin φ, cos θ)          avec r(θ, φ) = Σ c[l,m] · Y[l,m](θ, φ)".into());
                t.push("θ ∈ [0, π], φ ∈ [-π, π] : direction depuis le centre (colatitude depuis +z, longitude)".into());
            }
            t.extend([frame, String::new()]);
            if explicit {
                t.push("Harmoniques développées en cos θ, sin θ, cos mφ, sin mφ (normalisation incluse). Forme exacte, mais mal conditionnée".into());
                t.push("en virgule flottante au-delà de l ≈ 25 : pour un calcul numérique, préférer les récurrences (exports GLSL et Python).".into());
            } else {
                t.extend(["Harmoniques sphériques réelles orthonormées, phase de Condon-Shortley :", "  Y[l,0]  = N(l,0) · P_l^0(cos θ)", "  Y[l,m]  = √2 · N(l,m) · P_l^m(cos θ) · cos(mφ)          (m > 0)",
                    "  Y[l,-m] = √2 · N(l,m) · P_l^m(cos θ) · sin(mφ)          (m > 0)", "  N(l,m)  = √[ (2l+1)/(4π) · (l-m)!/(l+m)! ]          P_l^m : fonction de Legendre associée, avec le facteur (-1)^m"].map(String::from));
            }
            t.extend([String::new(), if b { "Σ c[l,m] · Y[l,m](θ, φ) =" } else { "r(θ, φ) =" }.to_string()]);
            let polys = if explicit { Sh::new(l).polys() } else { vec![] };
            for (k, v) in &r {
                let (ll, m) = (k[0] as usize, k[1]);
                let cs = if b { format!("+ ({}, {}, {})", num(v[0]), num(v[1]), num(v[2])) } else { num(v[0]) };
                t.push(if explicit { format!("  {cs} · {}          ← Y[{ll},{m}]", ylm_explicit(&polys[ll * (ll + 1) / 2 + m.unsigned_abs() as usize], m)) } else { format!("  {cs} · Y[{ll},{m}]") });
            }
        }
    }
    t
}

/// Shader GLSL évaluant la série (méthodes A et B).
pub fn glsl(md: &Model, o: &Outcome, l: usize, win: Window) -> String {
    let Fit::Sh(f) = &o.fit else {
        return format!("// HarmoMesh — méthode C, K = {l} : {} coefficients complexes.\n// Trop nombreux pour un tableau constant : charger le JSON dans une texture 3D\n// et sommer f(p) = Re Σ c_k exp(2πi k·u) dans le shader (surface : f = 0).\n", (2 * l + 1).pow(3));
    };
    let (ty, n) = (if f.nc == 1 { "float" } else { "vec3" }, (l + 1) * (l + 1));
    let vals: Vec<String> = rows(o, l, win).iter().map(|(_, v)| if v.len() == 1 { format!("{:.7e}", v[0]) } else { format!("vec3({:.7e},{:.7e},{:.7e})", v[0], v[1], v[2]) }).collect();
    format!(r#"// HarmoMesh — {name}, méthode {m}, L = {l} ({n} coefficients {ty}).
// Harmoniques réelles orthonormées, phase de Condon-Shortley, theta = colatitude (depuis +z).
// Repère normalisé : original = p * {scale:e} + vec3({cx:e}, {cy:e}, {cz:e}).
const int L = {l};
const {ty} C[{n}] = {ty}[{n}](
{vals});

{ty} harmo(float theta, float phi) {{
    float ct = cos(theta), st = sin(theta), pmm = 0.28209479;
    {ty} r = {ty}(0.0);
    for (int m = 0; m <= L; m++) {{
        float fm = float(m);
        if (m > 0) pmm *= -sqrt((2.0 * fm + 1.0) / (2.0 * fm)) * st;
        float cm = cos(fm * phi), sm = sin(fm * phi), p1 = pmm, p2 = 0.0;
        for (int l = m; l <= L; l++) {{
            float fl = float(l);
            if (l > m) {{
                float a = sqrt((4.0 * fl * fl - 1.0) / (fl * fl - fm * fm));
                float b = sqrt(max(((fl - 1.0) * (fl - 1.0) - fm * fm) / (4.0 * (fl - 1.0) * (fl - 1.0) - 1.0), 0.0));
                float p = a * (ct * p1 - b * p2); p2 = p1; p1 = p;
            }}
            int i = l * l + l;
            if (m == 0) r += C[i] * p1; else r += 1.41421356 * p1 * (C[i + m] * cm + C[i - m] * sm);
        }}
    }}
    return r;
}}

// Point de la surface pour (theta, phi), dans le repère normalisé.
vec3 harmoPoint(float theta, float phi) {{
    {point}
}}
"#, name = md.name, m = o.method.letter(), scale = md.xf.scale, cx = md.xf.center[0], cy = md.xf.center[1], cz = md.xf.center[2], vals = vals.join(",\n"),
        point = if f.nc == 1 { "return harmo(theta, phi) * vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));" } else { "return harmo(theta, phi);" })
}

/// Évaluateur Python autonome (NumPy) lisant le JSON exporté.
pub const PYTHON: &str = r#""""Évaluateur HarmoMesh (généré automatiquement). Usage : python harmomesh_eval.py coeffs.json"""
import json, sys
import numpy as np


def real_sh(L, theta, phi):
    """Harmoniques réelles orthonormées, phase de Condon-Shortley ; colonne l*l + l + m."""
    ct, st = np.cos(theta), np.sin(theta)
    Y = np.zeros((len(theta), (L + 1) ** 2))
    pmm = np.full_like(ct, 0.28209479177387814)
    for m in range(L + 1):
        if m:
            pmm = -np.sqrt((2 * m + 1) / (2 * m)) * st * pmm
        p1, p2 = pmm, 0 * ct
        for l in range(m, L + 1):
            if l > m:
                a = np.sqrt((4 * l * l - 1) / (l * l - m * m))
                b = np.sqrt(max(((l - 1) ** 2 - m * m) / (4 * (l - 1) ** 2 - 1), 0))
                p1, p2 = a * (ct * p1 - b * p2), p1
            i = l * l + l
            if m == 0:
                Y[:, i] = p1
            else:
                Y[:, i + m] = np.sqrt(2) * p1 * np.cos(m * phi)
                Y[:, i - m] = np.sqrt(2) * p1 * np.sin(m * phi)
    return Y


def evaluate(d, theta=None, phi=None, xyz=None):
    """A : rayon r(theta, phi) ; B : points (x, y, z)(theta, phi) ; C : f(x, y, z), surface f = 0.
    Entrées et sorties dans le repère d'origine du mesh."""
    c = np.array(d["coefficients"], float)
    s, o = d["normalization"]["scale"], np.array(d["normalization"]["center"])
    if d["method"] == "C":
        K, h = d["kmax"], d["half_size"]
        u = ((np.atleast_2d(xyz) - o) / s + h) / (2 * h)
        k = np.stack(np.meshgrid(*[np.arange(-K, K + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        return (np.exp(2j * np.pi * u @ k.T) @ (c[:, 0] + 1j * c[:, 1])).real * s
    Y = real_sh(d["lmax"], np.atleast_1d(np.asarray(theta, float)), np.atleast_1d(np.asarray(phi, float)))
    return (Y @ c) * s + (o if d["method"] == "B" else 0)


if __name__ == "__main__":
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    print("méthode", d["method"], "- degré", d["lmax"], "-", len(d["coefficients"]), "coefficients")
    if d["method"] == "C":
        print("f(centre) =", evaluate(d, xyz=[d["normalization"]["center"]]))
    else:
        print("valeur en (theta, phi) = (pi/2, 0) :", evaluate(d, [np.pi / 2], [0.0]))
"#;

/// Écrit tous les fichiers d'export dans `dir` ; renvoie leurs chemins.
pub fn export_all(dir: &Path, md: &Model, o: &Outcome, p: &Params, l: usize, win: Window) -> Result<Vec<PathBuf>, String> {
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    let stem = format!("{}_{}_L{l}", Path::new(&md.name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or("mesh".into()), o.method.letter());
    let mut out = vec![];
    let mut w = |name: String, data: String| -> Result<(), String> { let p = dir.join(name); std::fs::write(&p, data).map_err(|e| e.to_string())?; out.push(p); Ok(()) };
    w(format!("{stem}.json"), json(md, o, p, l, win).to_string())?;
    w(format!("{stem}.csv"), csv(o, l, win, ","))?;
    w(format!("{stem}.glsl"), glsl(md, o, l, win))?;
    w(format!("{stem}_formule.txt"), formula(md, o, l, win, false).join("\n"))?;
    w("harmomesh_eval.py".into(), PYTHON.into())?;
    let rec = o.mesh_at(l, win, None);
    for ext in ["ply", "obj"] { let p = dir.join(format!("{stem}.{ext}")); rec.save(&p, Some(md.xf))?; out.push(p); }
    Ok(out)
}
