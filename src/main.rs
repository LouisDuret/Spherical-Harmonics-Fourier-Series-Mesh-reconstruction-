//! HarmoMesh : sans argument, ouvre l'interface ; sinon ligne de commande (voir `harmomesh help`).
use harmomesh::{export, mesh::Mesh, pipeline::*, sh::Window};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[macro_use]
mod tex;
mod formula;
mod gui;

#[cfg(windows)]
extern "system" { fn FreeConsole() -> i32; }

const HELP: &str = "HarmoMesh — mesh 3D → séries de Fourier / harmoniques sphériques

  harmomesh [gui] [fichier]            interface graphique
  harmomesh diag  <fichier>            diagnostic topologique et méthodes disponibles
  harmomesh run   <fichier> [options]  chaîne complète + export
  harmomesh bench <fichier> [options]  tableau d'erreurs A / B / C (section 3 de la note)
  harmomesh gen   <sphere|ellipsoid|torus> <sortie.obj>

Options : -m A|B|C|auto   méthode (auto : arbre de décision)      -l N       degré L max (ou K pour C)
          -o DOSSIER      dossier d'export (run)                   --lambda X régularisation (B)
          --grid N        grille de distance signée (C, déf. 128)  --window none|lanczos|hann
          --voxels N      résolution de réparation (déf. 160)      --closing N  fermeture (voxels)
          --tris N        triangles après simplification           --no-area-fix  sans correction d'aire (B)
          --repair / --no-repair   forcer / interdire la réparation (auto par défaut)
Formats lus : OBJ, PLY, STL, glTF/GLB.";

struct Args { file: PathBuf, p: Params, rp: RepairParams, method: Option<Method>, out: PathBuf, repair: Option<bool>, lset: bool, wset: bool }

fn parse(a: &[String]) -> Result<Args, String> {
    let mut r = Args { file: PathBuf::new(), p: Params::default(), rp: RepairParams::default(), method: None, out: "export".into(), repair: None, lset: false, wset: false };
    let mut it = a.iter();
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or(format!("valeur manquante après {k}"));
        let num = |s: &String| s.parse::<f64>().map_err(|_| format!("nombre invalide : {s}"));
        match k.as_str() {
            "-m" => r.method = match val()?.to_uppercase().as_str() { "A" => Some(Method::A), "B" => Some(Method::B), "C" => Some(Method::C), _ => None },
            "-l" => { r.p.lmax = num(val()?)? as usize; r.lset = true; }
            "-o" => r.out = val()?.into(),
            "--lambda" => r.p.lambda = num(val()?)?,
            "--grid" => r.p.grid = num(val()?)? as usize,
            "--voxels" => r.rp.voxels = num(val()?)? as usize,
            "--closing" => r.rp.closing = num(val()?)? as usize,
            "--tris" => r.rp.tris = num(val()?)? as usize,
            "--window" => { r.p.window = match val()?.as_str() { "lanczos" => Window::Lanczos, "hann" => Window::Hann, _ => Window::None }; r.wset = true; }
            "--no-area-fix" => r.p.area_fix = false,
            "--area-passes" => { r.p.area_fix = true; r.p.area_passes = num(val()?)? as usize; }
            "--repair" => r.repair = Some(true),
            "--no-repair" => r.repair = Some(false),
            _ if r.file.as_os_str().is_empty() => r.file = k.into(),
            _ => return Err(format!("argument inconnu : {k}")),
        }
    }
    if r.file.as_os_str().is_empty() { return Err("fichier manquant".into()); }
    Ok(r)
}

fn prepare(a: &Args) -> Result<Model, String> {
    let t = Instant::now();
    let md = Model::load(&a.file)?;
    println!("Import      {} — {} ({:.2} s)", md.name, md.diag.summary(), t.elapsed().as_secs_f64());
    if a.repair == Some(false) || (a.repair.is_none() && md.diag.avail(Method::B).is_ok()) { return Ok(md); }
    let t = Instant::now();
    let r = md.repair(&a.rp, &Ctl::default())?;
    println!("Réparation  voxel {}³, fermeture {} → {} (référence : {} triangles, {:.2} s)", a.rp.voxels, a.rp.closing, r.diag.summary(), r.reference().0.f.len(), t.elapsed().as_secs_f64());
    Ok(r)
}

fn methods(md: &Model) {
    for m in Method::ALL {
        match md.diag.avail(m) { Ok(None) => println!("  {} : disponible", m.name()), Ok(Some(w)) => println!("  {} : disponible, avertissement : {w}", m.name()), Err(e) => println!("  {} : indisponible ({e})", m.name()) }
    }
    println!("  Méthode conseillée : {}", md.diag.recommend().map_or("réparer d'abord".into(), |m| m.name().to_string()));
}

fn cli(cmd: &str, rest: &[String]) -> Result<(), String> {
    if cmd == "gen" {
        let [shape, out] = rest else { return Err("usage : harmomesh gen <sphere|ellipsoid|torus> <sortie.obj>".into()) };
        let mut m = match shape.as_str() { "torus" => Mesh::torus(1.0, 0.35, 96, 48), _ => Mesh::icosphere(5) };
        if shape == "ellipsoid" { m.v.iter_mut().for_each(|p| *p = [p[0], p[1] * 0.7, p[2] * 0.45]); }
        return m.save(Path::new(out), None);
    }
    let mut a = parse(rest)?;
    let md = prepare(&a)?;
    match cmd {
        "diag" => methods(&md),
        "run" => {
            methods(&md);
            a.p.method = a.method.or(md.diag.recommend()).ok_or("aucune méthode applicable : utiliser --repair")?;
            if a.p.method == Method::C { if !a.lset { a.p.lmax = 24; } if !a.wset { a.p.window = Window::Lanczos; } }
            let o = compute(&md, &a.p, &Ctl::default())?;
            println!("Méthode     {}, L = {} : {} coefficients ({} Ko)", o.method.name(), o.lmax, o.ncoef(o.lmax), o.ncoef(o.lmax) * 8 / 1024);
            for w in &o.warnings { println!("  ⚠ {w}"); }
            if let Some(p) = &o.param { println!("Paramétrage {} triangle(s) inversé(s), distorsion d'aire p1 = {:.3}, max ×{:.0}", p.flipped, p.p1, p.dmax); }
            for (n, t) in &o.timings { println!("  {n:<34}{:>9.1} ms", t * 1e3); }
            let m = &o.metrics;
            println!("Erreurs     Chamfer {:.2} %, Hausdorff {:.2} % de la diagonale ; normales {:.1}° ; volume {:+.2} %, aire {:+.2} %", m.chamfer, m.hausdorff, m.normal_deg, m.dvol, m.darea);
            println!("Spectre     E_l = {}", o.energy.iter().take(12).map(|e| format!("{e:.1e}")).collect::<Vec<_>>().join(" "));
            for f in export::export_all(&a.out, &md, &o, &a.p, o.lmax, a.p.window)? { println!("Export      {}", f.display()); }
        }
        "bench" => {
            println!("\n{:<16}{:<10}{:>13}{:>14}{:>12}{:>10}", "Méthode", "Réglage", "Coefficients", "Erreur moy.", "Erreur max.", "Temps");
            for (m, name, levels, win) in [(Method::A, "A · radiale", [16, 32, 63], Window::None), (Method::B, "B · SPHARM", [20, 40, 60], Window::None), (Method::C, "C · Fourier 3D", [16, 24, 32], Window::Lanczos)] {
                if a.method.is_some_and(|x| x != m) { continue; }
                let t = Instant::now();
                let o = match compute(&md, &Params { method: m, lmax: levels[2], window: win, ..a.p.clone() }, &Ctl::default()) { Ok(o) => o, Err(e) => { println!("{name:<16}{e}"); continue; } };
                let dt = t.elapsed().as_secs_f64();
                for l in levels {
                    let t = Instant::now();
                    let e = harmomesh::metrics::compare(&o.mesh_at(l, win, None), &md).0;
                    println!("{:<16}{:<10}{:>13}{:>12.2} %{:>10.2} %{:>8.2} s", name, format!("{} = {l}", if m == Method::C { "K" } else { "L" }), o.ncoef(l), e.chamfer, e.hausdorff, if l == levels[2] { dt } else { t.elapsed().as_secs_f64() });
                }
            }
            println!("\nErreurs en % de la diagonale de la boîte englobante (moy. = Chamfer, max. = Hausdorff).");
        }
        _ => return Err(format!("commande inconnue : {cmd}\n\n{HELP}")),
    }
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = match args.first().map(|s| s.as_str()) {
        None => {
            // Lancé par double-clic : ne pas garder de console ouverte derrière la fenêtre.
            #[cfg(windows)]
            unsafe { FreeConsole(); }
            gui::run(None, None)
        }
        Some("gui") => gui::run(args.get(1).map(PathBuf::from), args.iter().position(|a| a == "--shot").and_then(|i| args.get(i + 1)).map(PathBuf::from)),
        Some("help" | "-h" | "--help") => { println!("{HELP}"); Ok(()) }
        Some(c @ ("diag" | "run" | "bench" | "gen")) => cli(c, &args[1..]),
        Some(f) if Path::new(f).is_file() => gui::run(Some(f.into()), None),
        Some(c) => Err(format!("commande inconnue : {c}\n\n{HELP}")),
    };
    if let Err(e) = r { eprintln!("Erreur : {e}"); std::process::exit(1); }
}
