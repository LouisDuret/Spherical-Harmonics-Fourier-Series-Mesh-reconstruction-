//! Interface graphique (maquette « HarmoMesh », section 7) : une fenêtre, huit zones.
//! Les calculs lourds tournent dans des threads de travail avec progression et annulation.
use eframe::egui::{self, pos2, vec2, Align2, Color32, FontId, Rect, RichText, Sense, Stroke};
use crate::{formula, tex};
use harmomesh::{export, mesh::*, methods::*, metrics::{self, Metrics}, pipeline::*, sh::*};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{atomic::Ordering, mpsc, Arc};
use std::time::Duration;

const BLUE: Color32 = Color32::from_rgb(33, 113, 214);
const GREEN: Color32 = Color32::from_rgb(34, 160, 100);
const ORANGE: Color32 = Color32::from_rgb(230, 150, 20);
const GRAY: Color32 = Color32::from_rgb(185, 185, 185);
const COPPER: Color32 = Color32::from_rgb(205, 133, 85);

enum Msg { Model(Model), Outcome(Outcome, Params), Fail(String) }
struct Job { rx: mpsc::Receiver<Msg>, ctl: Arc<Ctl>, what: &'static str }
#[derive(PartialEq, Clone, Copy)]
enum Tab { Side, Overlay, Sphere, Ylm }
enum Act { Open(Option<PathBuf>), Repair, Compute, Cancel, Export(&'static str), AreaFix }

pub fn run(file: Option<PathBuf>, shot: Option<PathBuf>) -> Result<(), String> {
    let opt = eframe::NativeOptions { viewport: egui::ViewportBuilder::default().with_inner_size([1500.0, 1000.0]).with_title("HarmoMesh — prototype"), ..Default::default() };
    eframe::run_native("HarmoMesh", opt, Box::new(move |cc| {
        cc.egui_ctx.set_theme(egui::Theme::Dark);
        tex::install_fonts(&cc.egui_ctx);
        let mut a = App::new();
        a.auto = shot.is_some();
        a.shot = shot;
        if let Some(f) = file { a.open(&cc.egui_ctx, f); }
        Ok(Box::new(a))
    })).map_err(|e| e.to_string())
}

struct App {
    raw: Option<Arc<Model>>, model: Option<Arc<Model>>, out: Option<Arc<Outcome>>, out_p: Option<Params>,
    p: Params, rp: RepairParams, lam: String,
    job: Option<Job>, log: Vec<(String, u8, String)>,
    l: usize, playing: bool, next_tick: f64, only: Option<usize>,
    rec: Option<Mesh>, err: Vec<f32>, met: Metrics, dirty: bool,
    cache: HashMap<(usize, u8), Mesh>, crx: Option<mpsc::Receiver<(usize, u8, Mesh)>>,
    tab: Tab, show_err: bool, show_orig: bool, edges: bool, show_sphere: bool,
    cam: (f32, f32, f32), sort: (usize, bool), filter: String,
    rows: Vec<(Vec<i64>, Vec<f64>)>, rows_key: (usize, u8, usize, usize, bool),
    ylm: (usize, i32), exported: Option<String>, shot: Option<PathBuf>, frames: u32, auto: bool,
    show_formula: bool, explicit: bool, frows: formula::Rows, fpolys: Vec<Vec<f64>>, frows_key: (usize, u8, usize), fwide: f32,
}

fn sep(n: usize) -> String {
    let s = n.to_string();
    s.chars().enumerate().flat_map(|(i, c)| if i > 0 && (s.len() - i) % 3 == 0 { vec![' ', c] } else { vec![c] }).collect()
}

fn lerp_col(a: [f32; 3], b: [f32; 3], t: f32) -> Color32 { Color32::from_rgb((a[0] + (b[0] - a[0]) * t) as u8, (a[1] + (b[1] - a[1]) * t) as u8, (a[2] + (b[2] - a[2]) * t) as u8) }
/// Carte d'écart signé à échelle fixe : rouge (−2 % de la diagonale) → blanc (0) → bleu (+2 %).
fn err_col(e: f32) -> Color32 {
    let t = (e / 2.0).clamp(-1.0, 1.0);
    if t < 0.0 { lerp_col([255.0; 3], [215.0, 40.0, 40.0], -t) } else { lerp_col([255.0; 3], [40.0, 90.0, 225.0], t) }
}

/// Rendu logiciel d'un mesh (projection orthographique, tri en profondeur, ombrage de Gouraud).
#[allow(clippy::too_many_arguments)]
fn paint(painter: &egui::Painter, rect: Rect, cam: (f32, f32, f32), v: &[V3], f: &[[u32; 3]], cols: &[Color32], fill: bool, edges: bool, cull: bool) {
    let ((sy, cy), (sp, cp)) = (cam.0.sin_cos(), cam.1.sin_cos());
    let r = [[cy, 0.0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]];
    let (s, c) = (cam.2 * rect.width().min(rect.height()) * 0.42, rect.center());
    let pv: Vec<[f32; 3]> = v.iter().map(|p| { let p = p.map(|x| x as f32); r.map(|a| a[0] * p[0] + a[1] * p[1] + a[2] * p[2]) }).collect();
    let mut nz = vec![[0f32; 3]; v.len()];
    let mut order: Vec<(f32, u32)> = Vec::with_capacity(f.len());
    for (i, t) in f.iter().enumerate() {
        let (a, b, d) = (pv[t[0] as usize], pv[t[1] as usize], pv[t[2] as usize]);
        let (u, w) = ([b[0] - a[0], b[1] - a[1], b[2] - a[2]], [d[0] - a[0], d[1] - a[1], d[2] - a[2]]);
        let n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        for &k in t { for q in 0..3 { nz[k as usize][q] += n[q]; } }
        if !cull || n[2] > 0.0 { order.push((a[2] + b[2] + d[2], i as u32)); }
    }
    order.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut m = egui::Mesh::default();
    for (i, p) in pv.iter().enumerate() {
        let mut n = nz[i];
        if n[2] < 0.0 { n = [-n[0], -n[1], -n[2]]; }
        let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt().max(1e-20);
        let sh = 0.32 + 0.68 * ((-0.35 * n[0] + 0.45 * n[1] + 0.82 * n[2]) / l).max(0.0);
        let b = cols[if cols.len() == 1 { 0 } else { i }];
        m.colored_vertex(pos2(c.x + p[0] * s, c.y - p[1] * s), Color32::from_rgb((b.r() as f32 * sh) as u8, (b.g() as f32 * sh) as u8, (b.b() as f32 * sh) as u8));
    }
    let line = Color32::from_rgba_unmultiplied(20, 20, 20, if fill { 70 } else { 150 });
    for (_, i) in order {
        let t = f[i as usize];
        if fill { m.add_triangle(t[0], t[1], t[2]); }
        if edges { for k in 0..3 {
            let (a, b) = (m.vertices[t[k] as usize].pos, m.vertices[t[(k + 1) % 3] as usize].pos);
            let d = (b - a).normalized().rot90() * 0.4;
            let n = m.vertices.len() as u32;
            for q in [a + d, a - d, b - d, b + d] { m.colored_vertex(q, line); }
            m.add_triangle(n, n + 1, n + 2);
            m.add_triangle(n, n + 2, n + 3);
        } }
    }
    painter.with_clip_rect(rect).add(egui::Shape::mesh(m));
}

impl App {
    fn new() -> App {
        let p = Params::default();
        App {
            raw: None, model: None, out: None, out_p: None, lam: format!("{:e}", p.lambda), p, rp: RepairParams::default(),
            job: None, log: vec![], l: 0, playing: false, next_tick: 0.0, only: None,
            rec: None, err: vec![], met: Metrics::default(), dirty: false, cache: HashMap::new(), crx: None,
            tab: Tab::Side, show_err: true, show_orig: true, edges: false, show_sphere: false,
            cam: (0.6, 0.2, 1.0), sort: (0, true), filter: String::new(), rows: vec![], rows_key: (usize::MAX, 0, 0, 0, true),
            ylm: (3, 2), exported: None, shot: None, frames: 0, auto: false,
            show_formula: true, explicit: false, frows: vec![], fpolys: vec![], frows_key: (usize::MAX, 0, 0), fwide: 0.0,
        }
    }

    fn say(&mut self, lvl: u8, s: impl Into<String>) { self.log.push((chrono::Local::now().format("%H:%M:%S").to_string(), lvl, s.into())); }

    fn spawn(&mut self, ctx: &egui::Context, what: &'static str, f: impl FnOnce(&Ctl) -> Result<Msg, String> + Send + 'static) {
        let (tx, rx) = mpsc::channel();
        let ctl = Arc::new(Ctl::default());
        let (c2, ctx2) = (ctl.clone(), ctx.clone());
        std::thread::spawn(move || { let _ = tx.send(f(&c2).unwrap_or_else(Msg::Fail)); ctx2.request_repaint(); });
        self.job = Some(Job { rx, ctl, what });
    }

    fn open(&mut self, ctx: &egui::Context, path: PathBuf) {
        self.spawn(ctx, "open", move |c| { c.step(0.3, "Import et diagnostic")?; Model::load(&path).map(Msg::Model) });
    }

    fn act(&mut self, ctx: &egui::Context, a: Act) {
        match a {
            Act::Open(Some(p)) => self.open(ctx, p),
            Act::Open(None) => if let Some(p) = rfd::FileDialog::new().add_filter("Mesh 3D", &["obj", "ply", "stl", "gltf", "glb"]).pick_file() { self.open(ctx, p) },
            Act::Repair => if let Some(md) = self.raw.clone() { let rp = self.rp.clone(); self.spawn(ctx, "repair", move |c| md.repair(&rp, c).map(Msg::Model)); },
            Act::Compute => if let Some(md) = self.model.clone() { let p = self.p.clone(); self.playing = false; self.spawn(ctx, "compute", move |c| compute(&md, &p, c).map(|o| Msg::Outcome(o, p))); },
            Act::Cancel => if let Some(j) = &self.job { j.ctl.cancel.store(true, Ordering::Relaxed); },
            Act::AreaFix => { self.p.area_fix = true; self.act(ctx, Act::Compute); }
            Act::Export(kind) => {
                let (Some(md), Some(o)) = (self.model.clone(), self.out.clone()) else { return };
                let (l, w) = (self.l, self.p.window);
                let stem = format!("{}_{}_L{l}", md.name.rsplit_once('.').map_or(md.name.as_str(), |x| x.0), o.method.letter());
                let res = if kind == "all" {
                    let Some(dir) = rfd::FileDialog::new().pick_folder() else { return };
                    export::export_all(&dir, &md, &o, &self.p, l, w).map(|f| format!("{} fichiers dans {}", f.len(), dir.display()))
                } else {
                    let Some(path) = rfd::FileDialog::new().set_file_name(if kind == "py" { "harmomesh_eval.py".to_string() } else { format!("{stem}.{kind}") }).save_file() else { return };
                    match kind {
                        "ply" | "obj" => o.mesh_at(l, w, None).save(&path, Some(md.xf)),
                        _ => std::fs::write(&path, match kind { "json" => export::json(&md, &o, &self.p, l, w).to_string(), "csv" => export::csv(&o, l, w, ","), "glsl" => export::glsl(&md, &o, l, w), "txt" => export::formula(&md, &o, l, w, self.explicit).join("\n"), _ => export::PYTHON.to_string() }).map_err(|e| e.to_string()),
                    }.map(|_| path.display().to_string())
                };
                match res { Ok(s) => { self.say(0, format!("Export : {s}")); self.exported = Some(kind.to_uppercase()); } Err(e) => self.say(2, format!("Export impossible : {e}")) }
            }
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let Some(j) = &self.job else { return };
        let Ok(m) = j.rx.try_recv() else { ctx.request_repaint_after(Duration::from_millis(80)); return };
        let what = j.what;
        self.job = None;
        match m {
            Msg::Model(md) => {
                let md = Arc::new(md);
                if what == "open" {
                    self.say(0, format!("Chargé {} — {}", md.name, md.diag.summary()));
                    if !md.diag.watertight() { self.say(1, "Surface ouverte ou en plusieurs morceaux : seul « Réparer » est proposé"); }
                    self.raw = Some(md.clone());
                    ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!("HarmoMesh — prototype · {}", md.name)));
                } else {
                    self.say(0, format!("Réparation voxel {}³ + fermeture morphologique → {}", self.rp.voxels, md.diag.summary()));
                }
                if let Some(m) = md.diag.recommend() { self.p.method = m; }
                for m in Method::ALL { if let Ok(Some(w)) = md.diag.avail(m) { self.say(1, format!("Méthode {} : {w}", m.letter())); } }
                (self.model, self.out, self.rec, self.only, self.exported) = (Some(md), None, None, None, None);
                self.cache.clear();
            }
            Msg::Outcome(o, p) => {
                if let Some(pi) = &o.param { self.say(0, format!("Paramétrage sphérique cMCF ({} it.) : {} triangle(s) inversé(s) · distorsion d'aire p1 = {:.3}", p.cmcf_iters, pi.flipped, pi.p1)); }
                self.say(0, format!("Décomposition {} terminée en {:.2} s : {} coefficients, Chamfer {:.2} %, Hausdorff {:.2} %", o.method.letter(), o.timings.iter().map(|t| t.1).sum::<f64>(), sep(o.ncoef(o.lmax)), o.metrics.chamfer, o.metrics.hausdorff));
                for w in o.warnings.clone() { self.say(1, w); }
                (self.l, self.out_p, self.dirty, self.only, self.exported) = (o.lmax, Some(p), true, None, None);
                self.out = Some(Arc::new(o));
                self.cache.clear();
            }
            Msg::Fail(e) => self.say(2, e),
        }
    }

    /// Met à jour la reconstruction affichée (somme partielle au degré courant) et ses erreurs.
    fn refresh(&mut self, ctx: &egui::Context) {
        if let Some(rx) = &self.crx {
            if let Ok((k, w, m)) = rx.try_recv() { self.cache.insert((k, w), m); self.crx = None; self.dirty = true; } else { ctx.request_repaint_after(Duration::from_millis(50)); }
        }
        if self.playing && !self.dirty {
            let t = ctx.input(|i| i.time);
            let top = self.out.as_ref().map_or(0, |o| o.lmax);
            if t >= self.next_tick { if self.l >= top { self.playing = false; } else { self.l += 1; self.dirty = true; self.next_tick = t + 0.12; } }
            ctx.request_repaint();
        }
        if !self.dirty { return; }
        let (Some(o), Some(md)) = (self.out.clone(), self.model.clone()) else { return };
        let (l, w) = (self.l, self.p.window);
        let rec = match &o.fit {
            Fit::Sh(_) => Some(o.mesh_at(l, w, self.only)),
            Fit::Fourier(_) => self.cache.get(&(l, w as u8)).cloned().or_else(|| {
                if self.crx.is_none() {
                    let (tx, rx) = mpsc::channel();
                    let (o2, ctx2) = (o.clone(), ctx.clone());
                    std::thread::spawn(move || { let _ = tx.send((l, w as u8, o2.mesh_at(l, w, None))); ctx2.request_repaint(); });
                    self.crx = Some(rx);
                }
                None
            }),
        };
        if let Some(r) = rec {
            (self.met, self.err) = metrics::compare(&r, &md);
            self.rec = Some(r);
            self.dirty = false;
        }
    }

    fn toolbar(&mut self, ui: &mut egui::Ui, act: &mut Vec<Act>) {
        let busy = self.job.is_some();
        if ui.add_enabled(!busy, egui::Button::new("Ouvrir…")).on_hover_text("OBJ, PLY, STL, glTF — ou glisser-déposer un fichier").clicked() { act.push(Act::Open(None)); }
        if ui.add_enabled(self.raw.is_some() && !busy, egui::Button::new("Réparer")).on_hover_text("Remaillage voxel étanche, puis simplification").clicked() { act.push(Act::Repair); }
        let white = |s: &str| RichText::new(s).strong().color(Color32::WHITE);
        if busy {
            if ui.add(egui::Button::new(white("Annuler")).fill(ORANGE)).clicked() { act.push(Act::Cancel); }
        } else {
            let ok = self.model.as_ref().is_some_and(|m| m.diag.avail(self.p.method).is_ok());
            if ui.add_enabled(ok, egui::Button::new(white("Calculer")).fill(BLUE)).clicked() { act.push(Act::Compute); }
        }
        ui.add_enabled_ui(self.out.is_some() && !busy, |ui| ui.menu_button("Exporter ⏷", |ui| {
            for (k, t) in [("all", "Tout, dans un dossier…"), ("json", "Coefficients JSON…"), ("csv", "Coefficients CSV…"), ("txt", "Formule complète (texte)…"), ("glsl", "Shader GLSL…"), ("py", "Évaluateur Python…"), ("ply", "Mesh reconstruit PLY…"), ("obj", "Mesh reconstruit OBJ…")] {
                if ui.button(t).clicked() { act.push(Act::Export(k)); ui.close_menu(); }
            }
        }));
        ui.toggle_value(&mut self.show_formula, "Formule").on_hover_text("Afficher ou masquer le panneau de la formule complète de l'objet");
        ui.separator();
        ui.label("Méthode");
        egui::ComboBox::from_id_salt("method").selected_text(self.p.method.name()).width(200.0).show_ui(ui, |ui| for m in Method::ALL {
            let (ok, why) = match self.model.as_ref().map(|md| md.diag.avail(m)) { Some(Err(e)) => (false, e), Some(Ok(Some(w))) => (true, w), _ => (true, String::new()) };
            let r = ui.add_enabled(ok, egui::SelectableLabel::new(self.p.method == m, m.name()));
            let r = if !ok { r.on_disabled_hover_text(why) } else if !why.is_empty() { r.on_hover_text(format!("⚠ {why}")) } else { r };
            if r.clicked() && self.p.method != m {
                if (m == Method::C) != (self.p.method == Method::C) { (self.p.lmax, self.p.window) = if m == Method::C { (24, Window::Lanczos) } else { (40, Window::None) }; self.dirty = true; }
                self.p.method = m;
            }
        });
        let c = self.p.method == Method::C;
        ui.label(if c { "K max" } else { "L max" });
        ui.add(egui::DragValue::new(&mut self.p.lmax).range(1..=if c { self.p.grid / 2 - 1 } else { 127 }));
        ui.add_enabled_ui(self.p.method == Method::B, |ui| {
            ui.label("Pondération");
            egui::ComboBox::from_id_salt("weight").selected_text(self.p.weight.name()).width(90.0).show_ui(ui, |ui| for w in [Weight::Area, Weight::Sphere, Weight::Uniform] { ui.selectable_value(&mut self.p.weight, w, w.name()); });
            ui.label("λ");
            if ui.add(egui::TextEdit::singleline(&mut self.lam).desired_width(56.0)).changed() { if let Ok(v) = self.lam.replace(',', ".").parse::<f64>() { if v >= 0.0 { self.p.lambda = v; } } }
        });
        if let Some(j) = &self.job { let (p, msg) = j.ctl.state.lock().unwrap().clone(); ui.add(egui::ProgressBar::new(p).text(msg).desired_width(280.0)); }
    }

    fn pipeline(&mut self, ui: &mut egui::Ui) {
        let running = self.job.as_ref().map(|j| j.what);
        let stale = self.out_p.as_ref().is_some_and(|q| Params { window: self.p.window, ..q.clone() } != self.p);
        let step = |ui: &mut egui::Ui, st: u8, title: &str, sub: String| {
            ui.horizontal(|ui| {
                let (r, p) = ui.allocate_painter(vec2(26.0, 34.0), Sense::hover());
                let (col, sym) = match st { 1 => (GREEN, "✔"), 2 => (ORANGE, "!"), 3 => (BLUE, "…"), _ => (GRAY, "") };
                p.circle_filled(r.rect.center(), 10.0, col);
                p.text(r.rect.center(), Align2::CENTER_CENTER, sym, FontId::proportional(13.0), Color32::WHITE);
                ui.vertical(|ui| { ui.label(RichText::new(title).strong().size(15.0)); ui.label(RichText::new(sub).weak()); });
            });
        };
        ui.add_space(6.0);
        ui.label(RichText::new("PIPELINE").strong().weak());
        let (raw, md, out) = (self.raw.clone(), self.model.clone(), self.out.clone());
        step(ui, if running == Some("open") { 3 } else if raw.is_some() { 1 } else { 0 }, "Import", raw.as_ref().map_or("aucun fichier".into(), |m| format!("{} sommets · {} tri.", sep(m.diag.nv), sep(m.diag.nf))));
        step(ui, raw.as_ref().map_or(0, |m| if m.diag.watertight() && m.diag.comps.len() == 1 { 1 } else { 2 }), "Diagnostic",
            raw.as_ref().map_or("—".into(), |m| format!("{} composante{} · {}", m.diag.comps.len(), if m.diag.comps.len() > 1 { "s" } else { "" }, if m.diag.watertight() { "étanche" } else { "ouvert" })));
        step(ui, if running == Some("repair") { 3 } else { md.as_ref().map_or(0, |m| if m.repaired || m.diag.watertight() { 1 } else { 0 }) }, "Réparation",
            md.as_ref().map_or("—".into(), |m| if m.repaired { format!("voxel {}³ · {}", self.rp.voxels, if m.diag.watertight() { "étanche" } else { "non étanche" }) } else if m.diag.watertight() { "non nécessaire".into() } else { "à lancer".into() }));
        egui::CollapsingHeader::new("Paramètres de réparation").show(ui, |ui| egui::Grid::new("rp").show(ui, |ui| {
            ui.label("Voxels (largeur)"); ui.add(egui::DragValue::new(&mut self.rp.voxels).range(32..=320)); ui.end_row();
            ui.label("Fermeture (voxels)"); ui.add(egui::DragValue::new(&mut self.rp.closing).range(0..=8)); ui.end_row();
            ui.label("Triangles cible"); ui.add(egui::DragValue::new(&mut self.rp.tris).range(500..=200_000).speed(100)); ui.end_row();
        }));
        step(ui, md.as_ref().map_or(0, |m| if !m.diag.watertight() { 0 } else if m.diag.genus() == Some(0) { 1 } else { 2 }), "Topologie",
            md.as_ref().filter(|m| m.diag.watertight()).map_or("—".into(), |m| match m.diag.genus() { Some(g) => format!("χ = {} · genre {g}", m.diag.comps[0].euler), None => format!("{} composantes fermées", m.diag.comps.len()) }));
        let pi = out.as_ref().and_then(|o| o.param.as_ref());
        step(ui, if stale { 0 } else { pi.map_or(0, |p| if p.flipped == 0 { 1 } else { 2 }) }, "Paramétrage",
            if self.p.method != Method::B { "méthode B uniquement".into() } else { pi.map_or("à calculer".into(), |p| format!("cMCF · {} triangle{s} inversé{s}", p.flipped, s = if p.flipped > 1 { "s" } else { "" })) });
        if self.p.method == Method::B {
            egui::CollapsingHeader::new("Paramètres du paramétrage").show(ui, |ui| egui::Grid::new("pp").show(ui, |ui| {
                ui.label("Itérations cMCF"); ui.add(egui::DragValue::new(&mut self.p.cmcf_iters).range(5..=300)); ui.end_row();
                ui.label("Pas δ"); ui.add(egui::DragValue::new(&mut self.p.cmcf_delta).range(1e-4..=0.1).speed(1e-4)); ui.end_row();
                ui.label("Correction d'aire"); ui.checkbox(&mut self.p.area_fix, ""); ui.end_row();
                ui.label("Passes d'égalisation"); ui.add(egui::DragValue::new(&mut self.p.area_passes).range(10..=1000)); ui.end_row();
                ui.label("Alignement degré 1"); ui.checkbox(&mut self.p.align, ""); ui.end_row();
            }));
        }
        step(ui, if running == Some("compute") { 3 } else if out.is_some() && !stale { 1 } else { 0 }, "Décomposition",
            out.as_ref().filter(|_| !stale).map_or(if stale { "paramètres modifiés : à recalculer".into() } else { "à calculer".to_string() }, |o| format!("{} = {} · {} coeff.", if o.method == Method::C { "K" } else { "L" }, o.lmax, sep(o.ncoef(o.lmax)))));
        egui::CollapsingHeader::new("Paramètres de décomposition").show(ui, |ui| egui::Grid::new("dp").show(ui, |ui| {
            ui.label("Fenêtre anti-Gibbs");
            egui::ComboBox::from_id_salt("win").selected_text(self.p.window.name()).show_ui(ui, |ui| for w in [Window::None, Window::Lanczos, Window::Hann] { if ui.selectable_value(&mut self.p.window, w, w.name()).changed() { self.dirty = true; } });
            ui.end_row();
            ui.label("Grille SDF (C)"); ui.add(egui::DragValue::new(&mut self.p.grid).range(32..=256)); ui.end_row();
        }));
        step(ui, if self.exported.is_some() { 1 } else { 0 }, "Export", self.exported.clone().map_or("JSON · CSV · GLSL · PLY".into(), |k| format!("{k} écrit")));
        ui.add_space(10.0);
        ui.label(RichText::new("AFFICHAGE").strong().weak());
        ui.checkbox(&mut self.show_err, "Carte d'erreur");
        ui.checkbox(&mut self.show_orig, "Maillage d'origine");
        ui.checkbox(&mut self.edges, "Arêtes");
        ui.checkbox(&mut self.show_sphere, "Sphère paramétrique");
    }

    fn spectrum(&mut self, ui: &mut egui::Ui, o: &Outcome) {
        let (resp, p) = ui.allocate_painter(vec2(ui.available_width(), 150.0), Sense::click());
        let r = Rect::from_min_max(resp.rect.min + vec2(38.0, 6.0), resp.rect.max - vec2(4.0, 16.0));
        let lg: Vec<f32> = o.energy.iter().map(|x| x.max(1e-300).log10() as f32).collect();
        let hi = lg.iter().cloned().fold(f32::MIN, f32::max).ceil();
        let lo = (hi - 7.0).max(lg.iter().cloned().fold(f32::MAX, f32::min).floor()).min(hi - 1.0);
        let (font, n) = (FontId::proportional(10.0), lg.len());
        for d in lo as i32..=hi as i32 {
            if (hi - lo) > 4.0 && d % 2 != 0 { continue; }
            let y = r.bottom() - (d as f32 - lo) / (hi - lo) * r.height();
            p.line_segment([pos2(r.left(), y), pos2(r.right(), y)], Stroke::new(0.5_f32, Color32::from_gray(70)));
            p.text(pos2(r.left() - 4.0, y), Align2::RIGHT_CENTER, format!("1e{d}"), font.clone(), Color32::GRAY);
        }
        let bw = r.width() / n as f32;
        for (l, v) in lg.iter().enumerate() {
            let h = ((v - lo) / (hi - lo)).clamp(0.01, 1.0) * r.height();
            let col = if self.only == Some(l) { ORANGE } else if l <= self.l { BLUE } else { GRAY };
            p.rect_filled(Rect::from_min_max(pos2(r.left() + l as f32 * bw + 0.5, r.bottom() - h), pos2(r.left() + (l + 1) as f32 * bw - 0.5, r.bottom())), 0.0, col);
            if l % (if n > 45 { 10 } else { 5 }) == 0 { p.text(pos2(r.left() + (l as f32 + 0.5) * bw, r.bottom() + 2.0), Align2::CENTER_TOP, l.to_string(), font.clone(), Color32::GRAY); }
        }
        let hov = resp.hover_pos().map(|q| ((q.x - r.left()) / bw) as usize).filter(|&l| l < n);
        if let Some(l) = hov {
            if resp.clicked() && o.method != Method::C { self.only = if self.only == Some(l) { None } else { Some(l) }; self.dirty = true; }
            resp.on_hover_text(format!("degré {l} : E = {:.3e}{}", o.energy[l], if o.method != Method::C { "\nclic : isoler la contribution de ce degré" } else { "" }));
        }
    }

    fn table(&mut self, ui: &mut egui::Ui, o: &Outcome) {
        let key = (self.l, self.p.window as u8, Arc::as_ptr(self.out.as_ref().unwrap()) as usize, self.sort.0, self.sort.1);
        if key != self.rows_key {
            self.rows = export::rows(o, self.l, self.p.window);
            let (c, asc) = self.sort;
            self.rows.sort_by(|a, b| { let v = |r: &(Vec<i64>, Vec<f64>)| if c < r.0.len() { r.0[c] as f64 } else { r.1[c - r.0.len()] }; let o = v(a).total_cmp(&v(b)); if asc { o } else { o.reverse() } });
            self.rows_key = key;
        }
        let head = export::header(o);
        let nk = self.rows.first().map_or(2, |r| r.0.len());
        let w = (ui.available_width() - 16.0) / head.len() as f32;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            for (i, h) in head.iter().enumerate() {
                let t = format!("{h}{}", if self.sort.0 == i { if self.sort.1 { " ⏶" } else { " ⏷" } } else { "" });
                if ui.add_sized([w, 18.0], egui::Button::new(RichText::new(t).strong()).frame(false)).on_hover_text("Trier").clicked() { self.sort = (i, if self.sort.0 == i { !self.sort.1 } else { true }); }
            }
        });
        let flt: Option<i64> = self.filter.trim().parse().ok();
        let rows: Vec<&(Vec<i64>, Vec<f64>)> = self.rows.iter().filter(|r| flt.is_none_or(|f| r.0[0] == f)).collect();
        egui::ScrollArea::vertical().max_height(170.0).auto_shrink([false, false]).show_rows(ui, 16.0, rows.len(), |ui, range| {
            for r in &rows[range] { ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                for k in &r.0 { ui.add_sized([w, 16.0], egui::Label::new(RichText::new(k.to_string()).monospace())); }
                for v in &r.1 { ui.add_sized([w, 16.0], egui::Label::new(RichText::new(format!("{v:+.3}")).monospace())); }
            }); }
        });
        ui.horizontal(|ui| {
            ui.label(format!("{} lignes", sep(rows.len())));
            ui.add(egui::TextEdit::singleline(&mut self.filter).desired_width(46.0).hint_text(if nk == 2 { "l = ?" } else { "kx = ?" })).on_hover_text("Filtrer sur la première colonne");
            if ui.link("Copier").clicked() { ui.ctx().copy_text(export::csv(o, self.l, self.p.window, "\t")); }
        });
    }

    fn measures(&self, ui: &mut egui::Ui, o: &Outcome) {
        egui::Grid::new("mes").num_columns(2).spacing([20.0, 3.0]).show(ui, |ui| {
            let mut row = |a: &str, b: String| { ui.label(a); ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(b).monospace())); ui.end_row(); };
            for (n, t) in &o.timings { row(n, if *t < 1.0 { format!("{:.0} ms", t * 1e3) } else { format!("{t:.2} s") }); }
            let top = if o.method == Method::C { "K" } else { "L" };
            row(&format!("Chamfer ({top} = {})", o.lmax), format!("{:.2} %", o.metrics.chamfer));
            row(&format!("Hausdorff ({top} = {})", o.lmax), format!("{:.2} %", o.metrics.hausdorff));
            row("Écart des normales", format!("{:.1}°", o.metrics.normal_deg));
            row("Volume / aire", format!("{:+.2} % / {:+.2} %", o.metrics.dvol, o.metrics.darea));
            if let Some(p) = &o.param { row("Distorsion d'aire p1", format!("{:.3}", p.p1)); }
            let b = o.ncoef(o.lmax) * 8;
            row("Mémoire coeff.", if b < 1 << 20 { format!("{} Ko", b / 1024) } else { format!("{:.1} Mo", b as f64 / (1 << 20) as f64) });
        });
    }

    fn view(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            for (t, n) in [(Tab::Side, "Côte à côte"), (Tab::Overlay, "Superposé"), (Tab::Sphere, "Sphère"), (Tab::Ylm, "Spectre 3D")] { ui.selectable_value(&mut self.tab, t, n); }
            if self.tab == Tab::Ylm {
                ui.separator();
                ui.add(egui::Slider::new(&mut self.ylm.0, 0..=24).text("l"));
                self.ylm.1 = self.ylm.1.clamp(-(self.ylm.0 as i32), self.ylm.0 as i32);
                ui.add(egui::Slider::new(&mut self.ylm.1, -(self.ylm.0 as i32)..=self.ylm.0 as i32).text("m"));
            }
        });
        let (resp, painter) = ui.allocate_painter(ui.available_size(), Sense::click_and_drag());
        let full = resp.rect;
        painter.rect_filled(full, 4.0, Color32::from_gray(20));
        if resp.dragged() { let d = resp.drag_delta(); self.cam.0 += d.x * 0.01; self.cam.1 = (self.cam.1 + d.y * 0.01).clamp(-1.57, 1.57); }
        if resp.hovered() { self.cam.2 = (self.cam.2 * (ui.input(|i| i.smooth_scroll_delta.y) * 0.003).exp()).clamp(0.2, 20.0); }
        if resp.double_clicked() { self.cam = (0.6, 0.2, 1.0); }
        let title = |r: Rect, s: String| { painter.text(r.center_top() + vec2(0.0, 8.0), Align2::CENTER_TOP, s, FontId::proportional(15.0), Color32::from_gray(210)); };
        let note = |s: &str| { painter.text(full.center(), Align2::CENTER_CENTER, s, FontId::proportional(15.0), Color32::GRAY); };
        let (md, out, cam) = (self.model.clone(), self.out.clone(), self.cam);
        let sphere_cols = |pi: &ParamInfo| -> Vec<Color32> { pi.distortion.iter().map(|d| { let t = (d.ln() / 100f64.ln()).clamp(-1.0, 1.0) as f32; if t >= 0.0 { lerp_col([245.0; 3], [200.0, 30.0, 30.0], t) } else { lerp_col([245.0; 3], [40.0, 80.0, 190.0], -t) } }).collect() };
        match self.tab {
            Tab::Ylm => {
                let (l, m) = self.ylm;
                let (dirs, pts, f) = latlon_grid(60, 120);
                let sh = Sh::new(l);
                let mut y = vec![0.0; sh.n()];
                let val: Vec<f64> = pts.iter().map(|&(ct, ph)| { sh.eval(ct, ph, &mut y); y[((l * l + l) as i64 + m as i64) as usize] }).collect();
                let top = val.iter().fold(1e-9f64, |a, b| a.max(b.abs()));
                let v: Vec<V3> = dirs.iter().zip(&val).map(|(d, y)| mul(*d, y.abs() / top)).collect();
                let cols: Vec<Color32> = val.iter().map(|y| err_col((2.0 * y / top) as f32)).collect();
                paint(&painter, full, cam, &v, &f, &cols, true, false, false);
                title(full, format!("Harmonique Y(l = {l}, m = {m}) — rouge : négatif, blanc : 0, bleu : positif"));
            }
            Tab::Sphere => match (&md, out.as_ref().and_then(|o| o.param.as_ref())) {
                (Some(md), Some(pi)) => {
                    paint(&painter, full, cam, &pi.sphere, &md.mesh.f, &sphere_cols(pi), true, true, true);
                    title(full, format!("Paramétrage sphérique — rouge : zones comprimées (max ×{:.1}), bleu : dilatées", pi.dmax));
                }
                _ => note("Le paramétrage sphérique n'existe que pour la méthode B (après calcul)."),
            },
            Tab::Side | Tab::Overlay => {
                let Some(md) = md else { note("Ouvrir un mesh (OBJ, PLY, STL, glTF) ou le glisser dans la fenêtre."); return };
                let cull = md.diag.watertight();
                let side = self.tab == Tab::Side && self.show_orig && out.is_some();
                let (left, right) = if side { full.split_left_right_at_fraction(0.5) } else { (full, full) };
                let orig_name = if md.repaired { "Original (réparé)" } else { "Original" };
                let pi = out.as_ref().and_then(|o| o.param.as_ref()).filter(|_| self.show_sphere);
                if side || out.is_none() {
                    match pi { Some(pi) => paint(&painter, left, cam, &pi.sphere, &md.mesh.f, &sphere_cols(pi), true, self.edges, true), None => paint(&painter, left, cam, &md.mesh.v, &md.mesh.f, &[COPPER], true, self.edges, cull) }
                    title(left, if pi.is_some() { "Sphère paramétrique (distorsion d'aire)".into() } else { orig_name.into() });
                }
                if let Some(o) = &out {
                    match &self.rec {
                        Some(r) => {
                            let cols: Vec<Color32> = if self.show_err && self.only.is_none() && self.err.len() == r.v.len() { self.err.iter().map(|&e| err_col(e)).collect() } else { vec![Color32::from_gray(225)] };
                            paint(&painter, right, cam, &r.v, &r.f, &cols, true, self.edges, true);
                            if self.tab == Tab::Overlay && self.show_orig { paint(&painter, right, cam, &md.mesh.v, &md.mesh.f, &[COPPER], false, true, cull); }
                        }
                        None => { painter.text(right.center(), Align2::CENTER_CENTER, "Reconstruction en cours…", FontId::proportional(15.0), Color32::GRAY); }
                    }
                    let top = if o.method == Method::C { "K" } else { "L" };
                    title(right, match self.only { Some(d) => format!("Contribution isolée du degré {d}"), None => format!("Reconstruction  {top} = {}{}", self.l, if self.tab == Tab::Overlay && self.show_orig { "  + original en fil de fer" } else { "" }) });
                    if self.show_err {
                        let bar = Rect::from_min_size(full.right_bottom() - vec2(330.0, 28.0), vec2(180.0, 12.0));
                        for i in 0..60 { painter.rect_filled(Rect::from_min_size(bar.min + vec2(i as f32 * 3.0, 0.0), vec2(3.0, 12.0)), 0.0, err_col(i as f32 / 15.0 - 2.0)); }
                        painter.text(bar.left_center() - vec2(8.0, 0.0), Align2::RIGHT_CENTER, "Écart à l'original   −2 %", FontId::proportional(12.0), Color32::from_gray(210));
                        painter.text(bar.right_center() + vec2(8.0, 0.0), Align2::LEFT_CENTER, "+2 % diag.", FontId::proportional(12.0), Color32::from_gray(210));
                    }
                }
            }
        }
    }

    /// Panneau « Formule complète » : forme générale et définitions à gauche, développement terme à terme
    /// à droite (liste virtualisée), le tout composé avec des symboles mathématiques.
    fn formula_panel(&mut self, ui: &mut egui::Ui, act: &mut Vec<Act>) {
        ui.add_space(4.0);
        let (Some(md), Some(o)) = (self.model.clone(), self.out.clone()) else {
            ui.label(RichText::new("FORMULE COMPLÈTE").strong().weak());
            ui.label(RichText::new("La formule mathématique de l'objet apparaît ici après le calcul.").weak());
            ui.allocate_space(ui.available_size());
            return;
        };
        let key = (self.l, self.p.window as u8, Arc::as_ptr(&o) as usize);
        if key != self.frows_key {
            self.frows = formula::prepare(&o, export::rows(&o, self.l, self.p.window));
            self.fpolys = if o.method == Method::C { vec![] } else { Sh::new(self.l).polys() };
            (self.frows_key, self.fwide) = (key, 0.0);
        }
        let (rows, polys, isc) = (std::mem::take(&mut self.frows), std::mem::take(&mut self.fpolys), o.method == Method::C);
        let fx = formula::Ctx { o: &o, rows: &rows, polys: &polys, explicit: self.explicit && !isc };
        let (nc, n, l) = (fx.ncomp(), rows.len(), self.l);
        ui.horizontal(|ui| {
            ui.label(RichText::new("FORMULE COMPLÈTE").strong().weak());
            ui.label(format!("{} = {l}  ·  {} termes", if isc { "K" } else { "L" }, sep(n * nc)));
            ui.separator();
            if !isc && ui.checkbox(&mut self.explicit, "Développer les harmoniques").changed() { self.fwide = 0.0; }
            if ui.button("Copier (LaTeX)").clicked() { ui.ctx().copy_text(fx.latex(&md, l)); }
            if ui.button("Copier (texte)").clicked() { ui.ctx().copy_text(export::formula(&md, &o, l, self.p.window, fx.explicit).join("\n")); }
            if ui.button("Enregistrer…").clicked() { act.push(Act::Export("txt")); }
        });
        ui.separator();
        let (col, h) = (ui.visuals().strong_text_color(), ui.available_height());
        ui.horizontal_top(|ui| {
            ui.allocate_ui_with_layout(vec2(440.0, h), egui::Layout::top_down(egui::Align::LEFT), |ui| {
                egui::ScrollArea::both().id_salt("fhead").auto_shrink([false, false]).show(ui, |ui| for (m, note) in fx.header(&md, l) {
                    if let Some(m) = m {
                        let f = tex::lay(ui.painter(), &m, 18.0, col);
                        let (rect, _) = ui.allocate_exact_size(vec2(f.w, f.up + f.down + 6.0), Sense::hover());
                        f.paint(ui.painter(), rect.left_top() + vec2(0.0, f.up + 3.0), col);
                    }
                    if !note.is_empty() { ui.label(RichText::new(note).weak()); }
                });
            });
            ui.separator();
            const RH: f32 = 26.0;
            const S: f32 = 16.0;
            let avail = ui.available_width();
            let (cellw, ncols) = if fx.explicit || isc { (0.0, 1) } else {
                let w = tex::lay(ui.painter(), &row![tex::r(" − "), formula::numm(0.5), tex::M::Sp(0.18), tex::subsup(tex::i("Y"), tex::r(l.to_string()), tex::r(format!("−{l}")))], S, col).w + 18.0;
                (w, ((avail - 40.0) / w).floor().max(1.0) as usize)
            };
            let rpc = 2 + n.div_ceil(ncols);
            let mut wide = self.fwide;
            ui.allocate_ui_with_layout(vec2(avail, h), egui::Layout::top_down(egui::Align::LEFT), |ui| egui::ScrollArea::both().id_salt("fterms").auto_shrink([false, false]).show_rows(ui, RH, rpc * nc, |ui, range| for idx in range {
                let (comp, rr) = (idx / rpc, idx % rpc);
                let (rect, _) = ui.allocate_exact_size(vec2(wide.max(avail - 24.0), RH), Sense::hover());
                let base = rect.left_top() + vec2(4.0, RH * 0.68);
                if rr == 0 {
                    tex::lay(ui.painter(), &fx.title(comp), S + 1.0, col).paint(ui.painter(), base, col);
                } else if rr < rpc - 1 {
                    for j in 0..ncols {
                        let t = (rr - 1) * ncols + j;
                        if t >= n { break; }
                        let (f, x) = (tex::lay(ui.painter(), &fx.term(comp, t), S, col), 28.0 + j as f32 * cellw);
                        f.paint(ui.painter(), base + vec2(x, 0.0), col);
                        wide = wide.max(x + f.w + 16.0);
                    }
                }
            }));
            self.fwide = wide;
        });
        (self.frows, self.fpolys) = (rows, polys);
    }

    fn slider(&mut self, ui: &mut egui::Ui) {
        let Some(o) = self.out.clone() else { ui.label(RichText::new("Le curseur de degré apparaît après le calcul.").weak()); return };
        ui.horizontal(|ui| {
            if ui.add(egui::Button::new(RichText::new(if self.playing { "⏸" } else { "▶" }).size(18.0).color(Color32::WHITE)).fill(BLUE).min_size(vec2(44.0, 30.0))).on_hover_text("Animer la construction progressive de la forme").clicked() {
                self.playing = !self.playing;
                if self.playing && self.l >= o.lmax { self.l = 0; self.dirty = true; }
            }
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("Degré affiché {}", if o.method == Method::C { "K" } else { "L" }));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(RichText::new(format!("{} coeff.  ·  Chamfer {:.2} %  ·  Hausdorff {:.2} %", sep(o.ncoef(self.l)), self.met.chamfer, self.met.hausdorff)).strong()));
                });
                ui.spacing_mut().slider_width = ui.available_width() - 60.0;
                if ui.add(egui::Slider::new(&mut self.l, 0..=o.lmax)).changed() { self.dirty = true; self.only = None; }
            });
        });
    }

    /// Mode automatique (`--shot fichier.png`) : enchaîne réparation et calcul, capture la fenêtre, puis quitte.
    fn selftest(&mut self, ctx: &egui::Context) {
        if !self.auto { return; }
        ctx.request_repaint_after(Duration::from_millis(50));
        let shot = ctx.input(|i| i.raw.events.iter().find_map(|e| if let egui::Event::Screenshot { image, .. } = e { Some(image.clone()) } else { None }));
        if let (Some(img), Some(path)) = (shot, &self.shot) {
            let r = std::fs::File::create(path).map_err(|e| e.to_string()).and_then(|f| {
                let mut enc = png::Encoder::new(std::io::BufWriter::new(f), img.size[0] as u32, img.size[1] as u32);
                enc.set_color(png::ColorType::Rgba);
                enc.write_header().and_then(|mut w| w.write_image_data(img.as_raw())).map_err(|e| e.to_string())
            });
            if let Err(e) = r { eprintln!("capture : {e}"); }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        if self.job.is_some() { return; }
        match (&self.model, &self.out) {
            (Some(m), None) if !m.diag.watertight() && !m.repaired => self.act(ctx, Act::Repair),
            (Some(m), None) if self.frames == 0 => {
                let env = |k: &str| std::env::var(k).unwrap_or_default();
                match env("HARMOMESH_METHOD").as_str() { "A" => self.p.method = Method::A, "C" => { self.p.method = Method::C; self.p.lmax = 20; self.p.window = Window::Lanczos; self.p.grid = 96; } _ => {} }
                self.tab = match env("HARMOMESH_TAB").as_str() { "overlay" => Tab::Overlay, "sphere" => Tab::Sphere, "ylm" => Tab::Ylm, _ => Tab::Side };
                self.edges = env("HARMOMESH_TAB") == "edges";
                self.explicit = env("HARMOMESH_TAB") == "formula2";
                if m.diag.avail(self.p.method).is_ok() { self.frames = 1; self.act(ctx, Act::Compute); }
            }
            (Some(_), Some(_)) if !self.dirty => { self.frames += 1; if self.frames == 12 { ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default())); } }
            (None, _) | (Some(_), None) if self.log.iter().any(|l| l.1 == 2) => { eprintln!("{}", self.log.last().unwrap().2); ctx.send_viewport_cmd(egui::ViewportCommand::Close); }
            _ => {}
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        self.poll(ctx);
        self.refresh(ctx);
        let mut act: Vec<Act> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).take(1).map(|p| Act::Open(Some(p))).collect());
        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| { ui.add_space(5.0); ui.horizontal(|ui| self.toolbar(ui, &mut act)); ui.add_space(5.0); });
        egui::TopBottomPanel::bottom("log").resizable(true).default_height(120.0).frame(egui::Frame::new().fill(Color32::from_gray(12)).inner_margin(egui::Margin::same(8))).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
                if self.log.is_empty() { ui.label(RichText::new("Journal — ouvrir un mesh pour commencer.").monospace().color(Color32::GRAY)); }
                for (t, lvl, s) in &self.log { ui.horizontal(|ui| {
                    let col = [Color32::from_gray(225), Color32::from_rgb(245, 170, 40), Color32::from_rgb(240, 90, 80)][*lvl as usize];
                    ui.label(RichText::new(format!("[{t}]  {}{s}", if *lvl > 0 { "⚠ " } else { "" })).monospace().color(col));
                    if *lvl == 1 && s.contains("envisager une correction d'aire") && self.job.is_none() && ui.small_button("Appliquer une correction d'aire").clicked() { act.push(Act::AreaFix); }
                }); }
            });
        });
        egui::SidePanel::left("pipeline").exact_width(250.0).show(ctx, |ui| egui::ScrollArea::vertical().show(ui, |ui| self.pipeline(ui)));
        egui::SidePanel::right("data").exact_width(340.0).show(ctx, |ui| egui::ScrollArea::vertical().show(ui, |ui| {
            let Some(o) = self.out.clone() else { ui.add_space(8.0); ui.label(RichText::new("Spectre, coefficients et mesures apparaissent après le calcul.").weak()); return };
            ui.add_space(6.0);
            ui.label(RichText::new(if o.method == Method::C { "SPECTRE D'ÉNERGIE PAR COUCHE |k|" } else { "SPECTRE D'ÉNERGIE PAR DEGRÉ" }).strong().weak());
            self.spectrum(ui, &o);
            ui.add_space(6.0);
            ui.label(RichText::new(if o.method == Method::C { "COEFFICIENTS  c(k)" } else { "COEFFICIENTS  c(l, m)" }).strong().weak());
            self.table(ui, &o);
            ui.add_space(6.0);
            ui.label(RichText::new("MESURES").strong().weak());
            self.measures(ui, &o);
        }));
        egui::TopBottomPanel::bottom("degree").show(ctx, |ui| { ui.add_space(6.0); self.slider(ui); ui.add_space(6.0); });
        if self.show_formula { egui::TopBottomPanel::bottom("formula").resizable(true).default_height(300.0).min_height(110.0).show(ctx, |ui| self.formula_panel(ui, &mut act)); }
        egui::CentralPanel::default().show(ctx, |ui| self.view(ui));
        for a in act { self.act(ctx, a); }
        self.selftest(ctx);
    }
}
