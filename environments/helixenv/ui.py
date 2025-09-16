from __future__ import annotations

"""
Helix Pro TUI (Textual)

An advanced terminal UI for Helix that wires presets from uses.md, shows
structured tables, and exports JSON summaries. Requires `textual`.
"""

from typing import Any, Dict, List, Optional

import importlib.util
import json
import os
import sys
from datetime import datetime


def _ensure_helix_on_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    code_path = os.path.join(root, "code")
    if code_path not in sys.path:
        sys.path.insert(0, code_path)


def _has_textual() -> bool:
    return importlib.util.find_spec("textual") is not None


def _pro_user_config_path() -> str:
    """Return a user config path for Pro TUI settings."""
    try:
        import platformdirs  # type: ignore
        cfg_dir = platformdirs.user_config_dir("helix", "helix")
    except Exception:
        cfg_dir = os.path.join(os.path.expanduser("~"), ".config", "helix")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "helix_pro_tui.json")


def _pro_save_config(cfg: Dict[str, Any]) -> None:
    try:
        p = _pro_user_config_path()
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, p)
    except Exception:
        pass


def _pro_load_config() -> Dict[str, Any]:
    try:
        with open(_pro_user_config_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _compute_pro_metrics(params: Dict[str, Any], which: Dict[str, bool], progress_cb=None) -> Dict[str, Any]:
    """Compute metrics according to selected presets. Returns JSONable dict."""
    _ensure_helix_on_path()
    import numpy as np
    from helix.cli import MLP, make_moons
    from helix import (
        extract_partitions,
        region_counts,
        mass_consistency_errors,
        build_V_from_incidence,
        sanity_check_ucp,
        ulam_pf,
        spectral_gap,
    )
    from helix.ktheory import k_invariants_from_B

    def step(msg: str):
        if progress_cb:
            progress_cb(msg)

    # Data + model
    step("Generating dataset…")
    X, y = make_moons(n=int(params.get("samples", 4000)), noise=float(params.get("noise", 0.07)), seed=int(params.get("seed", 1)))
    width = int(params.get("width", 16))
    epochs = int(params.get("epochs", 100))
    no_train = bool(params.get("no_train", False))

    step("Building model…")
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # pragma: no cover
        raise RuntimeError("PyTorch is required for Helix Pro TUI") from e

    model = MLP(d_in=2, widths=(width, width), d_out=2)
    if not no_train:
        step("Training model…")
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(epochs):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    # Extract partitions if any of region/mass/cp/k requested
    results: Dict[str, Any] = {}
    af = None
    if which.get("regions") or which.get("mass") or which.get("cp") or which.get("ktheory") or which.get("sparse"):
        step("Extracting partitions…")
        af = extract_partitions(model, X)

    if which.get("regions"):
        results["region_counts"] = region_counts(af.B_list)

    if which.get("mass"):
        results["mass_errors"] = mass_consistency_errors(af.B_list, af.tau_list)

    if which.get("cp"):
        step("Computing CP diagnostics…")
        cp_stats: List[Dict[str, float]] = []
        for k, B in enumerate(af.B_list, start=1):
            tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
            tau_cur = af.tau_list[k - 1]
            V = build_V_from_incidence(B, tau_prev, tau_cur)
            cp_stats.append(sanity_check_ucp(V, trials=6))
        results["cp_stats"] = cp_stats

    if which.get("ulam"):
        step("Running Ulam PF…")
        lo = np.array([-2.0, -1.5])
        hi = np.array([3.0, 2.5])
        # Use a small residual block map similar to CLI demo
        import torch
        import torch.nn as nn

        h2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
        with torch.no_grad():
            for p in h2.parameters():
                p.mul_(0.1)

        def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
            x_t = torch.from_numpy(x_np.astype(np.float32))
            y_ = x_t + eps * h2(x_t)
            return y_.detach().numpy().astype(np.float64)

        P, _ = ulam_pf(
            lambda z: F_block(z, eps=float(params.get("ulam_eps", 0.4))),
            (lo, hi),
            bins_per_dim=int(params.get("ulam_bins", 25)),
            samples_per_cell=int(params.get("ulam_samples_per_cell", 4)),
        )
        results["ulam_spectral_gap"] = spectral_gap(P)

    if which.get("ktheory"):
        step("Computing K-theory invariants…")
        k_list: List[Dict[str, Any]] = []
        for k, B in enumerate(af.B_list, start=1):
            try:
                inv = k_invariants_from_B(B)
                k_list.append({
                    "depth": k,
                    "rank": int(inv.get("rank", 0)),
                    "nullity": int(inv.get("nullity", 0)),
                    "torsion": list(map(int, inv.get("torsion", []))),
                    "S_diag": list(map(int, inv.get("S_diag", []))),
                })
            except Exception as e:
                k_list.append({"depth": k, "error": str(e)})
        results["k_invariants"] = k_list

    if which.get("sparse"):
        step("Summarizing sparse structure…")
        sparse: List[Dict[str, Any]] = []
        for k, B in enumerate(af.B_list, start=1):
            n_prev, n_cur = B.shape
            nnz = int(B.sum())
            parent_ptrs = int(len(af.parent_of_list[k - 1]))
            sparse.append({
                "depth": k,
                "shape": [n_prev, n_cur],
                "nnz": nnz,
                "parent_ptrs": parent_ptrs,
                "dense_bytes": int(n_prev * n_cur),
                "parent_bytes": int(parent_ptrs),
                "sparsity": float(1.0 - (nnz / max(1, n_prev * n_cur))),
            })
        results["sparse_stats"] = sparse

    results["config"] = params.copy()
    return results


def run_pro_tui(argv: Optional[List[str]] = None) -> int:
    if not _has_textual():
        print("Textual is not installed. Install: pip install textual")
        return 1

    _ensure_helix_on_path()
    from textual.app import App, ComposeResult
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        Log,
        ProgressBar,
        Checkbox,
        Static,
        DataTable,
    )
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual import on

    DEFAULTS = {
        "samples": 4000,
        "noise": 0.07,
        "seed": 1,
        "width": 16,
        "epochs": 100,
        "no_train": False,
        "ulam_bins": 25,
        "ulam_samples_per_cell": 4,
        "ulam_eps": 0.4,
        "export_dir": "",
    }

    class HelixProTUI(App):
        CSS = """
        Screen { layout: vertical; }
        #top { height: auto; }
        #left { width: 42; padding: 1 2; border: round $primary; }
        #center { padding: 1 2; border: round $panel; }
        #right { padding: 1 2; border: round $surface; }
        #buttons, #buttons2 { padding: 1; }
        #logbox { height: 10; border: round $accent; }
        #tables { height: 1fr; }
        DataTable { border: round $boost; height: auto; }
        #help { height: 12; border: round $accent; }
        """
        BINDINGS = [
            ("r", "run", "Run"),
            ("w", "sweep", "Sweep→Compare"),
            ("a", "add_cmp", "Add Compare"),
            ("m", "export_cmp", "Export Compare"),
            ("c", "clear_cmp", "Clear Compare"),
            ("v", "eval", "Run Eval"),
            ("e", "export", "Export JSON"),
            ("x", "export_csv", "Export CSVs"),
            ("b", "export_bundle", "Export Bundle"),
            ("k", "set_api", "Set API"),
            ("p", "apply_provider", "Apply Provider"),
            ("h", "help", "Help"),
            (":", "command", "Command"),
            ("s", "save", "Save Config"),
            ("d", "reset", "Reset Defaults"),
            ("q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            cfg = _pro_load_config()
            self.params: Dict[str, Any] = DEFAULTS.copy()
            self.params.update(cfg.get("params", {}))
            self.which = {
                "regions": bool(cfg.get("which", {}).get("regions", True)),
                "mass": bool(cfg.get("which", {}).get("mass", True)),
                "cp": bool(cfg.get("which", {}).get("cp", True)),
                "ulam": bool(cfg.get("which", {}).get("ulam", True)),
                "ktheory": bool(cfg.get("which", {}).get("ktheory", False)),
                "sparse": bool(cfg.get("which", {}).get("sparse", True)),
            }
            self._last: Optional[Dict[str, Any]] = None
            self._compare: List[Dict[str, Any]] = []
            self._last_eval_summary: Optional[Dict[str, Any]] = None
            self._api_mem = {
                "api_key_var": str(cfg.get("api", {}).get("api_key_var", "OPENAI_API_KEY")),
                "api_base_url": str(cfg.get("api", {}).get("api_base_url", "")),
                "api_model_var": str(cfg.get("api", {}).get("api_model_var", "OPENAI_MODEL")),
                "api_model": str(cfg.get("api", {}).get("api_model", "")),
                "provider": str(cfg.get("api", {}).get("provider", "openai")),
            }

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            with Horizontal(id="top"):
                with Vertical(id="left"):
                    yield Label("Use Cases")
                    yield Checkbox("1) Region growth", True, id="regions")
                    yield Checkbox("2) Mass consistency", True, id="mass")
                    yield Checkbox("3) CP checks", True, id="cp")
                    yield Checkbox("4) Ulam mixing", True, id="ulam")
                    yield Checkbox("5) Sparse structure", True, id="sparse")
                    yield Checkbox("7) K-theory (SymPy)", False, id="ktheory")
                    yield Static()
                    yield Label("Presets")
                    with Horizontal():
                        yield Button("Quick demo", id="preset_quick")
                        yield Button("High-res Ulam", id="preset_ulam")
                    with Horizontal():
                        yield Button("K-theory only", id="preset_k")
                    with Horizontal():
                        yield Button("Model Health Audit", id="preset_health")
                        yield Button("Architecture Compare", id="preset_compare")
                    with Horizontal():
                        yield Button("Monitoring Run", id="preset_monitor")
                    yield Static()
                    yield Label("Actions")
                    with Horizontal(id="buttons"):
                        yield Button("Run", id="run")
                        yield Button("Sweep→Compare", id="sweep")
                        yield Button("Export JSON", id="export", disabled=True)
                        yield Button("Export CSV", id="export_csv", disabled=True)
                        yield Button("Export Bundle", id="export_bundle", disabled=True)
                    with Horizontal(id="buttons2"):
                        yield Button("Add to Compare", id="add_cmp", disabled=True)
                        yield Button("Export Compare", id="export_cmp", disabled=True)
                        yield Button("Clear Compare", id="clear_cmp", disabled=True)
                        yield Button("Save Config", id="save")
                        yield Button("Reset", id="reset")
                        yield Button("Quit", id="quit")
                with Vertical(id="center"):
                    yield Label("Parameters")
                    with Horizontal():
                        yield Input(str(self.params["samples"]), placeholder="samples", id="samples")
                        yield Input(str(self.params["noise"]), placeholder="noise", id="noise")
                        yield Input(str(self.params["seed"]), placeholder="seed", id="seed")
                        yield Input(str(self.params["width"]), placeholder="width", id="width")
                        yield Input(str(self.params["epochs"]), placeholder="epochs", id="epochs")
                    with Horizontal():
                        yield Checkbox("no_train", value=bool(self.params["no_train"]), id="no_train")
                        yield Input(str(self.params["ulam_bins"]), placeholder="ulam_bins", id="ulam_bins")
                        yield Input(str(self.params["ulam_samples_per_cell"]), placeholder="ulam_samples_per_cell", id="ulam_samples_per_cell")
                        yield Input(str(self.params["ulam_eps"]), placeholder="ulam_eps", id="ulam_eps")
                    with Horizontal():
                        yield Input("1,2,3", placeholder="sweep_seeds", id="sweep_seeds")
                        yield Input("8,16,32", placeholder="sweep_widths", id="sweep_widths")
                        yield Input(self.params.get("export_dir", ""), placeholder="export_dir (optional)", id="export_dir")
                        yield Input("run1", placeholder="run_label", id="run_label")
                    with Horizontal():
                        yield Input(self._api_mem.get("api_key_var", "OPENAI_API_KEY"), placeholder="api_key_var", id="api_key_var")
                        yield Input("", placeholder="api_key (hidden)", id="api_key")
                        yield Input(self._api_mem.get("api_base_url", ""), placeholder="api_base_url (optional)", id="api_base_url")
                    with Horizontal():
                        yield Input(self._api_mem.get("api_model_var", "OPENAI_MODEL"), placeholder="api_model_var", id="api_model_var")
                        yield Input(self._api_mem.get("api_model", ""), placeholder="api_model (e.g., gpt-4o-mini, llama-3)", id="api_model")
                    with Horizontal():
                        yield Input(self._api_mem.get("provider", "openai"), placeholder="provider (openai|anthropic|openrouter|local|azure-openai)", id="provider")
                        yield Button("Apply Provider", id="apply_provider")
                    yield Label("Verifiers Eval — helixenv")
                    with Horizontal():
                        # Environment fixed to helixenv; mode and answer mode configurable
                        yield Static("env_id=helixenv")
                        yield Input("mcq", placeholder="eval_answer_mode (mcq|open)", id="eval_answer_mode")
                        yield Input("zero_shot", placeholder="eval_mode (zero_shot|agentic)", id="eval_mode")
                        yield Input("8", placeholder="eval_max_episodes", id="eval_max_episodes")
                    with Horizontal():
                        yield Checkbox("shuffle_options", value=True, id="eval_shuffle_options")
                        yield Checkbox("with_refusal", value=False, id="eval_with_refusal")
                        yield Checkbox("use_think", value=False, id="eval_use_think")
                        yield Input("42", placeholder="seed", id="eval_seed")
                        yield Input("10", placeholder="max_turns (agentic)", id="eval_max_turns")
                    with Horizontal():
                        yield Input("", placeholder="system_prompt (optional)", id="eval_system_prompt")
                        yield Checkbox("enforce_format", value=True, id="eval_enforce_format")
                        yield Button("Run Eval (vf-eval)", id="eval_btn")
                    yield Label("Expected Output Format")
                    yield Static("Set answer mode to see guidance.", id="eval_format")
                    with Horizontal():
                        yield Button("Eval: helixenv mcq@8", id="eval_preset_h8")
                        # BixBench preset removed per scope (helixenv only)
                with Vertical(id="right"):
                    yield Label("Results")
                    yield ProgressBar(total=6, id="progress")
                    with VerticalScroll(id="tables"):
                        yield DataTable(id="t_regions")
                        yield DataTable(id="t_mass")
                        yield DataTable(id="t_cp")
                        yield DataTable(id="t_ulam")
                        yield DataTable(id="t_sparse")
                        yield DataTable(id="t_k")
                        yield DataTable(id="t_eval")
                    yield Static("Help (press 'h' to toggle)")
                    with VerticalScroll(id="help"):
                        yield Static("Loading help…", id="help_text")
                    with Horizontal(id="cmdbar"):
                        yield Label(":")
                        yield Input(placeholder="run | sweep seeds=1,2 widths=8,16 | set samples=2000 | export bundle | preset health", id="cmdline")
                    yield Static("Compare")
                    yield DataTable(id="t_regions_cmp")
                    yield DataTable(id="t_mass_cmp")
                    yield DataTable(id="t_cp_unital_cmp")
                    yield DataTable(id="t_cp_coiso_cmp")
                    yield DataTable(id="t_ulam_cmp")
                    yield DataTable(id="t_sparse_sparsity_cmp")
                    yield DataTable(id="t_k_rank_cmp")
                    yield Log(id="logbox")
            yield Footer()

        def _read(self) -> None:
            def get_num(id_: str, cast):
                w = self.query_one(f"#{id_}")
                if isinstance(w, Input):
                    s = w.value or w.placeholder
                    return cast(s)
                raise KeyError(id_)

            def get_check(id_: str) -> bool:
                w = self.query_one(f"#{id_}")
                if isinstance(w, Checkbox):
                    return bool(w.value)
                return False

            self.params = {
                "samples": int(get_num("samples", int)),
                "noise": float(get_num("noise", float)),
                "seed": int(get_num("seed", int)),
                "width": int(get_num("width", int)),
                "epochs": int(get_num("epochs", int)),
                "no_train": get_check("no_train"),
                "ulam_bins": int(get_num("ulam_bins", int)),
                "ulam_samples_per_cell": int(get_num("ulam_samples_per_cell", int)),
                "ulam_eps": float(get_num("ulam_eps", float)),
                "export_dir": (self.query_one("#export_dir", Input).value or "").strip(),
                "run_label": (self.query_one("#run_label", Input).value or "run").strip(),
            }
            for k in list(self.which.keys()):
                self.which[k] = get_check(k)

        def _log(self, msg: str) -> None:
            self.query_one("#logbox", Log).write_line(msg)
            self.refresh()

        def _set_busy(self, busy: bool) -> None:
            self.query_one("#run", Button).disabled = busy
            has_last = self._last is not None
            self.query_one("#export", Button).disabled = busy or (not has_last)
            self.query_one("#export_csv", Button).disabled = busy or (not has_last)
            self.query_one("#export_bundle", Button).disabled = busy or (not has_last)
            self.query_one("#add_cmp", Button).disabled = busy or (not has_last)
            has_cmp = len(self._compare) > 0
            self.query_one("#export_cmp", Button).disabled = busy or (not has_cmp)
            self.query_one("#clear_cmp", Button).disabled = busy or (not has_cmp)

        def _progress(self, msg: str) -> None:
            bar = self.query_one("#progress", ProgressBar)
            bar.advance(1)
            self._log(msg)

        def action_run(self) -> None:
            self._read()
            self._set_busy(True)
            self._last = None
            bar = self.query_one("#progress", ProgressBar)
            try:
                bar.update(progress=0)
            except Exception:
                # Fallback for older/newer Textual versions
                try:
                    bar.progress = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
            self.query_one("#logbox", Log).clear()

            def work():
                try:
                    return _compute_pro_metrics(self.params, self.which, progress_cb=lambda m: self.call_from_thread(self._progress, m))
                except Exception as e:  # pragma: no cover
                    self.call_from_thread(self._log, f"Error: {e}")
                    return None

            def _populate_tables(res: Dict[str, Any]) -> None:
                # Helper to set up tables
                def setup(dt: DataTable, cols: List[str], rows: List[List[Any]]):
                    dt.clear(columns=True)
                    for c in cols:
                        dt.add_column(c)
                    for r in rows:
                        dt.add_row(*[str(x) for x in r])

                # Regions
                dt = self.query_one("#t_regions", DataTable)
                if "region_counts" in res:
                    setup(dt, ["depth", "regions"], [[i+1, v] for i, v in enumerate(res["region_counts"])])
                    dt.display = True
                else:
                    dt.display = False

                # Mass
                dt = self.query_one("#t_mass", DataTable)
                if "mass_errors" in res:
                    setup(dt, ["depth", "l1_error"], [[i+1, v] for i, v in enumerate(res["mass_errors"])])
                    dt.display = True
                else:
                    dt.display = False

                # CP stats
                dt = self.query_one("#t_cp", DataTable)
                if "cp_stats" in res:
                    rows = []
                    for i, s in enumerate(res["cp_stats"], start=1):
                        rows.append([i, s.get("unital_err_fro", 0.0), s.get("coisometry_err_fro", 0.0), s.get("psd_min_eig_violation", 0.0)])
                    setup(dt, ["depth", "unital", "coiso", "psd_vio"], rows)
                    dt.display = True
                else:
                    dt.display = False

                # Ulam
                dt = self.query_one("#t_ulam", DataTable)
                if "ulam_spectral_gap" in res:
                    setup(dt, ["metric", "value"], [["spectral_gap", f"{res['ulam_spectral_gap']:.4f}"]])
                    dt.display = True
                else:
                    dt.display = False

                # Sparse
                dt = self.query_one("#t_sparse", DataTable)
                if "sparse_stats" in res:
                    rows = []
                    for d in res["sparse_stats"]:
                        n_prev, n_cur = d.get("shape", [0, 0])
                        rows.append([d.get("depth", 0), n_prev, n_cur, d.get("nnz", 0), d.get("parent_ptrs", 0), f"{float(d.get('sparsity', 0.0)):.2%}"])
                    setup(dt, ["depth", "n_prev", "n_cur", "nnz", "parent_ptrs", "sparsity"], rows)
                    dt.display = True
                else:
                    dt.display = False

                # K-theory
                dt = self.query_one("#t_k", DataTable)
                if "k_invariants" in res:
                    rows = []
                    for ki in res["k_invariants"]:
                        if "error" in ki:
                            rows.append([ki.get("depth", 0), "error", ki.get("error", "")])
                        else:
                            rows.append([ki.get("depth", 0), ki.get("rank", 0), ki.get("nullity", 0), ",".join(map(str, ki.get("torsion", []))), ",".join(map(str, ki.get("S_diag", [])))])
                    # If errors present, columns differ; unify by using strings
                    setup(dt, ["depth", "rank", "nullity", "torsion", "S_diag"], rows)
                    dt.display = True
                else:
                    dt.display = False

            def done(res: Optional[Dict[str, Any]]):
                self._set_busy(False)
                if res is None:
                    return
                self._last = res
                self.query_one("#export", Button).disabled = False
                self.query_one("#export_csv", Button).disabled = False
                self.query_one("#export_bundle", Button).disabled = False
                self.query_one("#add_cmp", Button).disabled = False
                # Pretty print results
                if "region_counts" in res:
                    self._log(f"[AF] region counts: {res['region_counts']}")
                if "mass_errors" in res:
                    self._log(f"[AF] mass L1 errors: {res['mass_errors']}")
                if "cp_stats" in res:
                    for s in res["cp_stats"]:
                        self._log(
                            "[CP] unital={:.2e} coiso={:.2e} psd_vio={:.2e}".format(
                                s.get("unital_err_fro", 0.0), s.get("coisometry_err_fro", 0.0), s.get("psd_min_eig_violation", 0.0)
                            )
                        )
                if "ulam_spectral_gap" in res:
                    self._log(f"[ULAM] gap: {res['ulam_spectral_gap']:.4f}")
                if "sparse_stats" in res:
                    for d in res["sparse_stats"]:
                        self._log(f"[SPARSE] depth={d['depth']} shape={tuple(d['shape'])} nnz={d['nnz']} parents={d['parent_ptrs']} sparsity={d['sparsity']:.2%}")
                if "k_invariants" in res:
                    for ki in res["k_invariants"]:
                        if "error" in ki:
                            self._log(f"[K] depth={ki['depth']} error={ki['error']}")
                        else:
                            self._log(f"[K] depth={ki['depth']} rank={ki['rank']} nullity={ki['nullity']} torsion={ki['torsion']}")
                _populate_tables(res)
                self._populate_compare_tables()

            # Use plain threading to avoid Textual Worker API differences
            import threading
            def _runner():
                res = work()
                self.call_from_thread(done, res)
            threading.Thread(target=_runner, daemon=True, name="helix-pro-run").start()

        def _resolve_dir(self) -> str:
            d = (self.params.get("export_dir") or "").strip()
            if not d:
                return os.getcwd()
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                return os.getcwd()
            return d

        def action_export(self) -> None:
            if not self._last:
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            d = self._resolve_dir()
            fn = os.path.join(d, f"helix_pro_metrics_{ts}.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(self._last, f, indent=2)
            self._log(f"Exported {os.path.abspath(fn)}")

        def action_export_csv(self) -> None:
            if not self._last:
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            prefix = f"helix_pro_{ts}_"
            base = self._resolve_dir()
            def w(name: str, header: List[str], rows: List[List[Any]]):
                path = os.path.join(base, prefix + name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                self._log(f"Exported {os.path.abspath(path)}")

            res = self._last
            # Regions
            if "region_counts" in res:
                w("regions.csv", ["depth", "regions"], [[i+1, v] for i, v in enumerate(res["region_counts"])])
            # Mass
            if "mass_errors" in res:
                w("mass.csv", ["depth", "l1_error"], [[i+1, v] for i, v in enumerate(res["mass_errors"])])
            # CP
            if "cp_stats" in res:
                rows = []
                for i, s in enumerate(res["cp_stats"], start=1):
                    rows.append([i, s.get("unital_err_fro", 0.0), s.get("coisometry_err_fro", 0.0), s.get("psd_min_eig_violation", 0.0)])
                w("cp.csv", ["depth", "unital", "coiso", "psd_vio"], rows)
            # Ulam
            if "ulam_spectral_gap" in res:
                w("ulam.csv", ["metric", "value"], [["spectral_gap", f"{res['ulam_spectral_gap']:.4f}"]])
            # Sparse
            if "sparse_stats" in res:
                rows = []
                for d in res["sparse_stats"]:
                    n_prev, n_cur = d.get("shape", [0, 0])
                    rows.append([
                        d.get("depth", 0),
                        n_prev,
                        n_cur,
                        d.get("nnz", 0),
                        d.get("parent_ptrs", 0),
                        f"{float(d.get('sparsity', 0.0)):.4%}",
                    ])
                w("sparse.csv", ["depth", "n_prev", "n_cur", "nnz", "parent_ptrs", "sparsity"], rows)
            # K theory
            if "k_invariants" in res:
                rows = []
                for ki in res["k_invariants"]:
                    if "error" in ki:
                        rows.append([ki.get("depth", 0), "error", ki.get("error", ""), "", ""])
                    else:
                        rows.append([ki.get("depth", 0), ki.get("rank", 0), ki.get("nullity", 0), ";".join(map(str, ki.get("torsion", []))), ";".join(map(str, ki.get("S_diag", [])))])
                w("ktheory.csv", ["depth", "rank", "nullity", "torsion", "S_diag"], rows)

        def action_export_bundle(self) -> None:
            if not self._last:
                self._log("Nothing to export yet.")
                return
            import zipfile, tempfile
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = self._resolve_dir()
            bundle = os.path.join(base_dir, f"helix_bundle_{ts}.zip")
            tmpdir = tempfile.mkdtemp(prefix="helix_bundle_")

            # Save config (excluding secret key)
            cfg = {
                "params": self.params,
                "which": self.which,
                "api": {
                    "api_key_var": (self.query_one("#api_key_var", Input).value or self._api_mem.get("api_key_var", "")),
                    "api_base_url": (self.query_one("#api_base_url", Input).value or self._api_mem.get("api_base_url", "")),
                    "api_model_var": (self.query_one("#api_model_var", Input).value or self._api_mem.get("api_model_var", "")),
                    "api_model": (self.query_one("#api_model", Input).value or self._api_mem.get("api_model", "")),
                    "provider": (self.query_one("#provider", Input).value or self._api_mem.get("provider", "")),
                },
            }
            with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            # Metrics JSON
            with open(os.path.join(tmpdir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(self._last, f, indent=2)

            # Helper to write CSVs
            def wcsv(name: str, header: List[str], rows: List[List[Any]]):
                p = os.path.join(tmpdir, name)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")

            res = self._last
            if "region_counts" in res:
                wcsv("regions.csv", ["depth", "regions"], [[i+1, v] for i, v in enumerate(res["region_counts"])])
            if "mass_errors" in res:
                wcsv("mass.csv", ["depth", "l1_error"], [[i+1, v] for i, v in enumerate(res["mass_errors"])])
            if "cp_stats" in res:
                rows = []
                for i, s in enumerate(res["cp_stats"], start=1):
                    rows.append([i, s.get("unital_err_fro", 0.0), s.get("coisometry_err_fro", 0.0), s.get("psd_min_eig_violation", 0.0)])
                wcsv("cp.csv", ["depth", "unital", "coiso", "psd_vio"], rows)
            if "ulam_spectral_gap" in res:
                wcsv("ulam.csv", ["metric", "value"], [["spectral_gap", f"{res['ulam_spectral_gap']:.4f}"]])
            if "sparse_stats" in res:
                rows = []
                for d in res["sparse_stats"]:
                    n_prev, n_cur = d.get("shape", [0, 0])
                    rows.append([d.get("depth", 0), n_prev, n_cur, d.get("nnz", 0), d.get("parent_ptrs", 0), f"{float(d.get('sparsity', 0.0)):.4%}"])
                wcsv("sparse.csv", ["depth", "n_prev", "n_cur", "nnz", "parent_ptrs", "sparsity"], rows)
            if "k_invariants" in res:
                rows = []
                for ki in res["k_invariants"]:
                    if "error" in ki:
                        rows.append([ki.get("depth", 0), "error", ki.get("error", "")])
                    else:
                        rows.append([ki.get("depth", 0), ki.get("rank", 0), ki.get("nullity", 0), ";".join(map(str, ki.get("torsion", []))), ";".join(map(str, ki.get("S_diag", [])))])
                wcsv("ktheory.csv", ["depth", "rank", "nullity", "torsion", "S_diag"], rows)

            # Compare outputs
            if self._compare:
                comp = {"runs": self._compare}
                if self._last_eval_summary is not None:
                    comp["eval_summary"] = self._last_eval_summary
                with open(os.path.join(tmpdir, "compare.json"), "w", encoding="utf-8") as f:
                    json.dump(comp, f, indent=2)
                labels = [c.get("label", f"run{i+1}") for i, c in enumerate(self._compare)]
                maxd = max((len(c["results"].get("region_counts", []) or []) for c in self._compare), default=0)
                rows = []
                for dpt in range(maxd):
                    row = [dpt+1]
                    for c in self._compare:
                        arr = c["results"].get("region_counts", []) or []
                        row.append(arr[dpt] if dpt < len(arr) else "")
                    rows.append(row)
                wcsv("compare_regions.csv", ["depth"] + labels, rows)
                row = [["spectral_gap"] + [
                    (f"{c['results'].get('ulam_spectral_gap', 0.0):.4f}" if c["results"].get("ulam_spectral_gap") is not None else "")
                    for c in self._compare
                ]]
                wcsv("compare_ulam.csv", ["metric"] + labels, row)

            import zipfile
            with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(tmpdir):
                    for fn in files:
                        p = os.path.join(root, fn)
                        zf.write(p, os.path.relpath(p, tmpdir))
            self._log(f"Exported bundle {os.path.abspath(bundle)}")

        def action_eval(self) -> None:
            # Run verifiers eval via vf-eval and show a summary table
            self._log("Starting verifiers eval…")
            env_id = "helixenv"
            answer_mode = (self.query_one("#eval_answer_mode", Input).value or "mcq").strip()
            mode = (self.query_one("#eval_mode", Input).value or "zero_shot").strip()
            try:
                max_eps = int((self.query_one("#eval_max_episodes", Input).value or "8").strip())
            except Exception:
                max_eps = 8
            # helixenv-specific args
            try:
                seed = int((self.query_one("#eval_seed", Input).value or "42").strip())
            except Exception:
                seed = 42
            try:
                max_turns = int((self.query_one("#eval_max_turns", Input).value or "10").strip())
            except Exception:
                max_turns = 10
            shuffle_options = self.query_one("#eval_shuffle_options", Checkbox).value  # type: ignore[attr-defined]
            with_refusal = self.query_one("#eval_with_refusal", Checkbox).value  # type: ignore[attr-defined]
            use_think = self.query_one("#eval_use_think", Checkbox).value  # type: ignore[attr-defined]
            sys_prompt_in = (self.query_one("#eval_system_prompt", Input).value or "").strip()
            env_args = {
                "mode": mode,
                "answer_mode": answer_mode,
                "max_episodes": max_eps,
                "shuffle_options": bool(shuffle_options),
                "with_refusal": bool(with_refusal),
                "use_think": bool(use_think),
                "seed": seed,
                "max_turns": max_turns,
            }

            # Optional enforcement of output format by setting system_prompt
            try:
                enforce = self.query_one("#eval_enforce_format", Checkbox).value  # type: ignore[attr-defined]
            except Exception:
                enforce = False
            if sys_prompt_in:
                env_args["system_prompt"] = sys_prompt_in
            elif enforce:
                env_args["system_prompt"] = self._compose_format_prompt(env_id, answer_mode, env_args)

            # Update format guidance UI
            self._update_eval_format_ui(env_id, answer_mode, env_args)

            import shutil as _sh
            exe = _sh.which("vf-eval")
            if not exe:
                self._log("vf-eval not found on PATH. Install verifiers or activate the env providing it.")
                return

            import subprocess
            cmd = [exe, env_id, "-a", json.dumps(env_args), "-s"]
            self._log("$ " + " ".join(cmd))

            dt = self.query_one("#t_eval", DataTable)
            dt.clear(columns=True)
            for c in ["metric", "value"]:
                dt.add_column(c)
            dt.display = True

            def work():
                try:
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    summary = {}
                    assert proc.stdout is not None
                    for line in proc.stdout:
                        self.call_from_thread(self._log, line.rstrip())
                        if "reward/avg=" in line:
                            try:
                                parts = line.strip().split()
                                for p in parts:
                                    if "=" in p:
                                        k, v = p.split("=", 1)
                                        summary[k] = v
                            except Exception:
                                pass
                    proc.wait()
                    # Remember last eval summary and context
                    self._last_eval_summary = {
                        "env_id": env_id,
                        "env_args": env_args,
                        "metrics": summary,
                        "returncode": proc.returncode,
                    }
                    for k, v in summary.items():
                        self.call_from_thread(dt.add_row, k, v)
                    return proc.returncode == 0
                except Exception as e:
                    self.call_from_thread(self._log, f"Eval error: {e}")
                    return False

            def done(ok: bool):
                self._log("Eval completed." if ok else "Eval failed.")

            import threading
            def _runner():
                ok = work()
                self.call_from_thread(done, ok)
            threading.Thread(target=_runner, daemon=True, name="helix-pro-eval").start()

        @on(Button.Pressed, "#eval_preset_h8")
        def _on_eval_preset_h8(self) -> None:
            # helixenv mcq@8 zero_shot
            self.query_one("#eval_answer_mode", Input).value = "mcq"  # type: ignore[attr-defined]
            self.query_one("#eval_mode", Input).value = "zero_shot"  # type: ignore[attr-defined]
            self.query_one("#eval_max_episodes", Input).value = "8"  # type: ignore[attr-defined]
            self._update_eval_format_ui("helixenv", "mcq", {"mode": "zero_shot"})
            self._log("Applied eval preset: helixenv mcq@8")

        # Removed bixbench preset per 'helixenv only' scope

        def _populate_compare_tables(self) -> None:
            # Build side-by-side tables across added runs
            def setup(dt: DataTable, cols: List[str], rows: List[List[Any]]):
                dt.clear(columns=True)
                for c in cols:
                    dt.add_column(c)
                for r in rows:
                    dt.add_row(*[str(x) for x in r])

            labels = [c.get("label", f"run{i+1}") for i, c in enumerate(self._compare)]
            ids = ["#t_regions_cmp", "#t_mass_cmp", "#t_cp_unital_cmp", "#t_cp_coiso_cmp", "#t_ulam_cmp", "#t_sparse_sparsity_cmp", "#t_k_rank_cmp"]
            if not labels:
                for id_ in ids:
                    if self.query(id_):
                        self.query_one(id_, DataTable).display = False
                return

            # Regions compare
            dt = self.query_one("#t_regions_cmp", DataTable)
            maxd = 0
            lists: List[List[Any]] = []
            for c in self._compare:
                arr = c["results"].get("region_counts", []) or []
                lists.append(list(arr))
                maxd = max(maxd, len(arr))
            rows = []
            for d in range(maxd):
                row = [d+1]
                for arr in lists:
                    row.append(arr[d] if d < len(arr) else "")
                rows.append(row)
            setup(dt, ["depth"] + labels, rows)
            dt.display = True

            # Mass compare
            dt = self.query_one("#t_mass_cmp", DataTable)
            lists = []
            maxd = 0
            for c in self._compare:
                arr = c["results"].get("mass_errors", []) or []
                lists.append(list(arr))
                maxd = max(maxd, len(arr))
            rows = []
            for d in range(maxd):
                row = [d+1]
                for arr in lists:
                    row.append(arr[d] if d < len(arr) else "")
                rows.append(row)
            setup(dt, ["depth"] + labels, rows)
            dt.display = True

            # CP unital/coiso compare
            def cp_rows(key: str) -> List[List[Any]]:
                outs: List[List[float]] = []
                maxd = 0
                for c in self._compare:
                    stats = c["results"].get("cp_stats", []) or []
                    arr = [s.get(key, 0.0) for s in stats]
                    outs.append(arr)
                    maxd = max(maxd, len(arr))
                rows: List[List[Any]] = []
                for d in range(maxd):
                    row = [d+1]
                    for arr in outs:
                        row.append(arr[d] if d < len(arr) else "")
                    rows.append(row)
                return rows

            dt = self.query_one("#t_cp_unital_cmp", DataTable)
            setup(dt, ["depth"] + labels, cp_rows("unital_err_fro"))
            dt.display = True
            dt = self.query_one("#t_cp_coiso_cmp", DataTable)
            setup(dt, ["depth"] + labels, cp_rows("coisometry_err_fro"))
            dt.display = True

            # Ulam compare
            dt = self.query_one("#t_ulam_cmp", DataTable)
            row = [["spectral_gap"] + [
                (f"{c['results'].get('ulam_spectral_gap', 0.0):.4f}" if c["results"].get("ulam_spectral_gap") is not None else "")
                for c in self._compare
            ]]
            setup(dt, ["metric"] + labels, row)
            dt.display = True

            # Sparse compare (sparsity by depth)
            dt = self.query_one("#t_sparse_sparsity_cmp", DataTable)
            maxd = 0
            lists = []
            for c in self._compare:
                stats = c["results"].get("sparse_stats", []) or []
                arr = [float(s.get("sparsity", 0.0)) for s in stats]
                lists.append(arr)
                maxd = max(maxd, len(arr))
            rows = []
            for d in range(maxd):
                row = [d+1]
                for arr in lists:
                    row.append(f"{arr[d]:.2%}" if d < len(arr) else "")
                rows.append(row)
            setup(dt, ["depth"] + labels, rows)
            dt.display = True

            # K-theory compare (rank by depth)
            dt = self.query_one("#t_k_rank_cmp", DataTable)
            maxd = 0
            lists = []
            for c in self._compare:
                items = c["results"].get("k_invariants", []) or []
                arr: List[Any] = []
                for it in items:
                    arr.append(it.get("rank", "err" if "error" in it else 0))
                lists.append(arr)
                maxd = max(maxd, len(arr))
            rows = []
            for d in range(maxd):
                row = [d+1]
                for arr in lists:
                    row.append(arr[d] if d < len(arr) else "")
                rows.append(row)
            setup(dt, ["depth"] + labels, rows)
            dt.display = True

        @on(Button.Pressed, "#add_cmp")
        def _on_add_cmp(self) -> None:
            if not self._last:
                return
            label = (self.params.get("run_label") or f"run{len(self._compare)+1}").strip()
            self._compare.append({"label": label, "results": self._last})
            self._log(f"Added run to compare: {label}")
            self._populate_compare_tables()
            self._set_busy(False)

        @on(Button.Pressed, "#clear_cmp")
        def _on_clear_cmp(self) -> None:
            self._compare.clear()
            self._log("Cleared comparison runs.")
            self._populate_compare_tables()
            self._set_busy(False)

        @on(Button.Pressed, "#export_cmp")
        def _on_export_cmp(self) -> None:
            if not self._compare:
                return
            d = self._resolve_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Export combined JSON
            out = {"runs": self._compare}
            fn = os.path.join(d, f"helix_compare_{ts}.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            self._log(f"Exported {os.path.abspath(fn)}")
            # Export CSVs for key compares
            def w(name: str, header: List[str], rows: List[List[Any]]):
                path = os.path.join(d, f"helix_compare_{ts}_" + name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                self._log(f"Exported {os.path.abspath(path)}")
            labels = [c.get("label", f"run{i+1}") for i, c in enumerate(self._compare)]
            # Regions
            maxd = max((len(c["results"].get("region_counts", []) or []) for c in self._compare), default=0)
            rows = []
            for dpt in range(maxd):
                row = [dpt+1]
                for c in self._compare:
                    arr = c["results"].get("region_counts", []) or []
                    row.append(arr[dpt] if dpt < len(arr) else "")
                rows.append(row)
            w("regions.csv", ["depth"] + labels, rows)
            # Ulam
            row = [["spectral_gap"] + [
                (f"{c['results'].get('ulam_spectral_gap', 0.0):.4f}" if c["results"].get("ulam_spectral_gap") is not None else "")
                for c in self._compare
            ]]
            w("ulam.csv", ["metric"] + labels, row)

        def action_reset(self) -> None:
            self.params = DEFAULTS.copy()
            self.query_one("#samples", Input).value = str(self.params["samples"])  # type: ignore[attr-defined]
            self.query_one("#noise", Input).value = str(self.params["noise"])  # type: ignore[attr-defined]
            self.query_one("#seed", Input).value = str(self.params["seed"])  # type: ignore[attr-defined]
            self.query_one("#width", Input).value = str(self.params["width"])  # type: ignore[attr-defined]
            self.query_one("#epochs", Input).value = str(self.params["epochs"])  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(self.params["ulam_samples_per_cell"])  # type: ignore[attr-defined]
            self.query_one("#ulam_eps", Input).value = str(self.params["ulam_eps"])  # type: ignore[attr-defined]
            for k, v in {"regions": True, "mass": True, "cp": True, "ulam": True, "ktheory": False, "sparse": True}.items():
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self._log("Parameters reset.")

        def action_quit(self) -> None:
            self.exit()

        # Format guidance helpers
        def _compose_format_prompt(self, env_id: str, answer_mode: str, env_args: Dict[str, Any]) -> str:
            env = env_id.strip().lower()
            mode = str(answer_mode or "mcq").strip().lower()
            qsrc = str(env_args.get("question_source", "")).strip().lower()
            if env == "helixenv":
                if mode == "mcq":
                    return (
                        "You are answering multiple‑choice questions about Helix. "
                        "Output exactly one letter A, B, C, or D. If an 'E. I don't know' option is shown, you may answer E. "
                        "Do not include any extra text."
                    )
                else:
                    return (
                        "Answer succinctly in one short sentence without qualifiers like 'I think'."
                    )
            if env == "bixbench":
                if qsrc == "hypothesis":
                    if mode == "mcq":
                        return (
                            "You are judging a hypothesis as True or False. "
                            "Output exactly one letter A or B (A=True, B=False). If an 'E. I don't know' option is shown, you may answer E."
                        )
                    else:
                        return "Output exactly one token: True or False."
                else:
                    if mode == "mcq":
                        return (
                            "You are answering a multiple‑choice scientific question. Output exactly one letter A, B, C, or D. "
                            "If an 'E. I don't know' option is shown, you may answer E."
                        )
                    else:
                        return (
                            "Answer the scientific question accurately in one short sentence."
                        )
            if env == "hle":
                # Generic guidance for HLE variants
                if mode == "mcq":
                    return "Output exactly one letter A–D (or E if 'I don't know' appears)."
                else:
                    return "Provide a concise short answer without extra commentary."
            # Fallback
            return "Follow the task instructions. Keep outputs minimal and strictly formatted."

        def _update_eval_format_ui(self, env_id: str, answer_mode: str, env_args: Dict[str, Any]) -> None:
            fmt = self._compose_format_prompt(env_id, answer_mode, env_args)
            try:
                box = self.query_one("#eval_format", Static)
                box.update(fmt)
            except Exception:
                pass

        # Preset buttons
        @on(Button.Pressed, "#preset_quick")
        def _on_preset_quick(self) -> None:
            # Fast run: no training, low Ulam res
            self.params.update({
                "samples": 2000,
                "noise": 0.07,
                "epochs": 0,
                "no_train": True,
                "ulam_bins": 16,
                "ulam_samples_per_cell": 1,
                "ulam_eps": 0.4,
            })
            for k, v in {"regions": True, "mass": True, "cp": True, "ulam": True, "sparse": True, "ktheory": False}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            # reflect inputs
            self.query_one("#samples", Input).value = str(self.params["samples"])  # type: ignore[attr-defined]
            self.query_one("#epochs", Input).value = str(self.params["epochs"])  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(self.params["ulam_samples_per_cell"])  # type: ignore[attr-defined]
            self._log("Applied preset: Quick demo")

        @on(Button.Pressed, "#preset_ulam")
        def _on_preset_ulam(self) -> None:
            # High resolution Ulam focus
            self.params.update({
                "ulam_bins": 48,
                "ulam_samples_per_cell": 4,
                "ulam_eps": 0.4,
                "epochs": 50,
                "no_train": True,
            })
            for k, v in {"regions": False, "mass": False, "cp": False, "ulam": True, "sparse": False, "ktheory": False}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(self.params["ulam_samples_per_cell"])  # type: ignore[attr-defined]
            self._log("Applied preset: High-res Ulam")

        @on(Button.Pressed, "#preset_k")
        def _on_preset_k(self) -> None:
            # K-theory only (requires sympy)
            self.params.update({
                "epochs": 0,
                "no_train": True,
            })
            for k, v in {"regions": False, "mass": False, "cp": False, "ulam": False, "sparse": False, "ktheory": True}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self._log("Applied preset: K-theory only")

        @on(Button.Pressed, "#preset_health")
        def _on_preset_health(self) -> None:
            # Model Health Audit
            self.params.update({
                "ulam_bins": 24,
                "ulam_samples_per_cell": 2,
                "ulam_eps": 0.4,
                "epochs": max(int(self.params.get("epochs", 50)), 50),
            })
            for k, v in {"regions": True, "mass": True, "cp": True, "ulam": True, "sparse": True, "ktheory": False}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(self.params["ulam_samples_per_cell"])  # type: ignore[attr-defined]
            self._log("Applied preset: Model Health Audit")

        @on(Button.Pressed, "#preset_compare")
        def _on_preset_compare(self) -> None:
            # Architecture Compare: configure sweeps
            self.params.update({
                "epochs": 0,
                "no_train": True,
            })
            self.query_one("#sweep_seeds", Input).value = "1,2,3"  # type: ignore[attr-defined]
            self.query_one("#sweep_widths", Input).value = "8,16,32"  # type: ignore[attr-defined]
            for k, v in {"regions": True, "mass": True, "cp": True, "ulam": True, "sparse": True, "ktheory": False}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self._log("Applied preset: Architecture Compare — click Sweep→Compare")

        @on(Button.Pressed, "#preset_monitor")
        def _on_preset_monitor(self) -> None:
            # Monitoring Run: fast JSON-friendly run
            self.params.update({
                "epochs": 0,
                "no_train": True,
                "ulam_bins": 16,
                "ulam_samples_per_cell": 1,
            })
            for k, v in {"regions": True, "mass": True, "cp": False, "ulam": False, "sparse": True, "ktheory": False}.items():
                self.which[k] = v
                self.query_one(f"#{k}", Checkbox).value = v  # type: ignore[attr-defined]
            self._log("Applied preset: Monitoring Run")

        # Compare hotkeys map to actions
        def action_add_cmp(self) -> None:
            if not self._last:
                return
            label = (self.params.get("run_label") or f"run{len(self._compare)+1}").strip()
            self._compare.append({"label": label, "results": self._last})
            self._log(f"Added run to compare: {label}")
            self._populate_compare_tables()
            self._set_busy(False)

        def action_clear_cmp(self) -> None:
            self._compare.clear()
            self._log("Cleared comparison runs.")
            self._populate_compare_tables()
            self._set_busy(False)

        def action_export_cmp(self) -> None:
            if not self._compare:
                return
            d = self._resolve_dir()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Export combined JSON
            out = {"runs": self._compare}
            if self._last_eval_summary is not None:
                out["eval_summary"] = self._last_eval_summary
            fn = os.path.join(d, f"helix_compare_{ts}.json")
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            self._log(f"Exported {os.path.abspath(fn)}")
            # Export CSVs for key compares
            def w(name: str, header: List[str], rows: List[List[Any]]):
                path = os.path.join(d, f"helix_compare_{ts}_" + name)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                self._log(f"Exported {os.path.abspath(path)}")
            labels = [c.get("label", f"run{i+1}") for i, c in enumerate(self._compare)]
            # Regions
            maxd = max((len(c["results"].get("region_counts", []) or []) for c in self._compare), default=0)
            rows = []
            for dpt in range(maxd):
                row = [dpt+1]
                for c in self._compare:
                    arr = c["results"].get("region_counts", []) or []
                    row.append(arr[dpt] if dpt < len(arr) else "")
                rows.append(row)
            w("regions.csv", ["depth"] + labels, rows)
            # Ulam
            row = [["spectral_gap"] + [
                (f"{c['results'].get('ulam_spectral_gap', 0.0):.4f}" if c["results"].get("ulam_spectral_gap") is not None else "")
                for c in self._compare
            ]]
            w("ulam.csv", ["metric"] + labels, row)

        # Sweeps -> Compare
        def action_sweep(self) -> None:
            seeds_str = self.query_one("#sweep_seeds", Input).value or "1,2,3"
            widths_str = self.query_one("#sweep_widths", Input).value or "8,16,32"
            try:
                seeds = [int(x.strip()) for x in seeds_str.split(",") if x.strip()]
            except Exception:
                seeds = [1, 2, 3]
            try:
                widths = [int(x.strip()) for x in widths_str.split(",") if x.strip()]
            except Exception:
                widths = [8, 16, 32]

            self._read()
            base = self.params.copy()
            base["epochs"] = int(base.get("epochs", 0))
            self._set_busy(True)
            self._log(f"Starting sweep over seeds={seeds} widths={widths}")

            def work():
                try:
                    for s in seeds:
                        for w in widths:
                            p = base.copy()
                            p["seed"] = s
                            p["width"] = w
                            res = _compute_pro_metrics(p, self.which, progress_cb=lambda m: self.call_from_thread(self._log, f"[{s},{w}] {m}"))
                            self._compare.append({"label": f"s{s}-w{w}", "results": res})
                            self.call_from_thread(self._log, f"Completed s={s} w={w}")
                    return True
                except Exception as e:
                    self.call_from_thread(self._log, f"Sweep error: {e}")
                    return False

            def done(ok: bool):
                self._set_busy(False)
                self._populate_compare_tables()
                self._log("Sweep completed." if ok else "Sweep failed.")

            import threading
            def _runner():
                ok = work()
                self.call_from_thread(done, ok)
            threading.Thread(target=_runner, daemon=True, name="helix-pro-sweep").start()

        # API key/model setter
        def action_set_api(self) -> None:
            var = (self.query_one("#api_key_var", Input).value or "").strip() or "OPENAI_API_KEY"
            key = (self.query_one("#api_key", Input).value or "").strip()
            base = (self.query_one("#api_base_url", Input).value or "").strip()
            model_var = (self.query_one("#api_model_var", Input).value or "").strip() or "OPENAI_MODEL"
            model_name = (self.query_one("#api_model", Input).value or "").strip()
            try:
                if key:
                    os.environ[var] = key
                if base:
                    # Set common base URL envs for OpenAI-compatible clients
                    os.environ["OPENAI_BASE_URL"] = base
                    os.environ["OPENAI_API_BASE"] = base
                    # Try provider-specific base var derived from key var prefix
                    if var.endswith("_API_KEY"):
                        prefix = var[:-len("_API_KEY")]
                        if prefix in ("OPENROUTER", "ANTHROPIC"):
                            os.environ[f"{prefix}_BASE_URL"] = base
                if model_name:
                    os.environ[model_var] = model_name
                msg = (f"Set {var} (hidden)" if key else "")
                if base:
                    msg += " and base URL"
                if model_name:
                    msg += f"; set {model_var}={model_name}"
                self._log(msg or "Updated API settings.")
            except Exception as e:
                self._log(f"Failed to set API key: {e}")

        def action_apply_provider(self) -> None:
            prov = (self.query_one("#provider", Input).value or "openai").strip().lower()
            if prov in ("openai", "azure-openai"):
                self.query_one("#api_key_var", Input).value = "OPENAI_API_KEY"  # type: ignore[attr-defined]
                self.query_one("#api_model_var", Input).value = "OPENAI_MODEL"  # type: ignore[attr-defined]
                if prov == "azure-openai":
                    self.query_one("#api_base_url", Input).value = "https://{resource}.openai.azure.com/openai/deployments/{deployment}/"  # type: ignore[attr-defined]
                else:
                    self.query_one("#api_base_url", Input).value = ""  # type: ignore[attr-defined]
            elif prov == "anthropic":
                self.query_one("#api_key_var", Input).value = "ANTHROPIC_API_KEY"  # type: ignore[attr-defined]
                self.query_one("#api_base_url", Input).value = "https://api.anthropic.com"  # type: ignore[attr-defined]
                self.query_one("#api_model_var", Input).value = "ANTHROPIC_MODEL"  # type: ignore[attr-defined]
            elif prov == "openrouter":
                self.query_one("#api_key_var", Input).value = "OPENROUTER_API_KEY"  # type: ignore[attr-defined]
                self.query_one("#api_base_url", Input).value = "https://openrouter.ai/api/v1"  # type: ignore[attr-defined]
                self.query_one("#api_model_var", Input).value = "OPENAI_MODEL"  # type: ignore[attr-defined]
            elif prov == "local":
                self.query_one("#api_key_var", Input).value = "OPENAI_API_KEY"  # type: ignore[attr-defined]
                self.query_one("#api_base_url", Input).value = "http://localhost:8000/v1"  # type: ignore[attr-defined]
                self.query_one("#api_model_var", Input).value = "OPENAI_MODEL"  # type: ignore[attr-defined]
            else:
                self._log(f"Unknown provider '{prov}'. Use openai|anthropic|openrouter|local|azure-openai.")
                return
            self._api_mem.update({
                "provider": prov,
                "api_key_var": self.query_one("#api_key_var", Input).value,
                "api_base_url": self.query_one("#api_base_url", Input).value,
                "api_model_var": self.query_one("#api_model_var", Input).value,
                "api_model": self.query_one("#api_model", Input).value,
            })
            self._log(f"Applied provider preset: {prov}")

        def action_save(self) -> None:
            # Persist settings except secret key
            self._read()
            cfg = {
                "params": self.params,
                "which": self.which,
                "api": {
                    "api_key_var": (self.query_one("#api_key_var", Input).value or self._api_mem.get("api_key_var", "")),
                    "api_base_url": (self.query_one("#api_base_url", Input).value or self._api_mem.get("api_base_url", "")),
                    "api_model_var": (self.query_one("#api_model_var", Input).value or self._api_mem.get("api_model_var", "")),
                    "api_model": (self.query_one("#api_model", Input).value or self._api_mem.get("api_model", "")),
                    "provider": (self.query_one("#provider", Input).value or self._api_mem.get("provider", "")),
                },
                "saved_at": datetime.utcnow().isoformat() + "Z",
            }
            _pro_save_config(cfg)
            self._log("Config saved.")

        def action_help(self) -> None:
            # Toggle help panel and populate from docs
            try:
                cont = self.query_one("#help")
                cont.display = not getattr(cont, "display", True)
            except Exception:
                pass
            try:
                txt = self.query_one("#help_text", Static)
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
                paths = [os.path.join(root, "docs", "applications.md"), os.path.join(root, "uses.md")]
                buf = []
                for p in paths:
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            buf.append(f"# {os.path.basename(p)}\n\n" + f.read())
                    except Exception:
                        pass
                if buf:
                    txt.update("\n\n".join(buf))
            except Exception:
                pass

        def action_command(self) -> None:
            try:
                self.query_one("#cmdline", Input).focus()
            except Exception:
                pass

        def _cmd_set(self, key: str, val: str) -> None:
            k = key.strip().lower()
            v = val.strip()
            try:
                numeric_int = {"samples", "seed", "width", "epochs", "ulam_bins", "ulam_samples_per_cell"}
                numeric_float = {"noise", "ulam_eps"}
                toggles = {"no_train", "regions", "mass", "cp", "ulam", "sparse", "ktheory"}
                misc_inputs = {"export_dir", "run_label", "api_key_var", "api_base_url", "api_model_var", "api_model", "provider"}
                if k in numeric_int:
                    self.query_one(f"#{k}", Input).value = str(int(float(v)))  # type: ignore[attr-defined]
                elif k in numeric_float:
                    self.query_one(f"#{k}", Input).value = str(float(v))  # type: ignore[attr-defined]
                elif k in toggles:
                    cb = self.query_one(f"#{k}", Checkbox)
                    cb.value = v.lower() in {"1", "true", "yes", "y", "on"}
                elif k in misc_inputs:
                    self.query_one(f"#{k}", Input).value = v  # type: ignore[attr-defined]
                else:
                    self._log(f"Unknown key: {k}")
                    return
                self._log(f"set {k}={v}")
            except Exception as e:
                self._log(f"set error: {e}")

        def _cmd_toggle(self, key: str) -> None:
            try:
                cb = self.query_one(f"#{key}", Checkbox)
                cb.value = not bool(cb.value)
                self._log(f"toggled {key} -> {cb.value}")
            except Exception:
                self._log(f"unknown toggle: {key}")

        def _cmd_exec(self, line: str) -> None:
            s = (line or "").strip()
            if not s:
                return
            low = s.lower()
            if low in {"run", "r"}:
                self.action_run()
                return
            if low in {"export", "export json", "e"}:
                self.action_export()
                return
            if low in {"export csv", "x"}:
                self.action_export_csv()
                return
            if low in {"export bundle", "bundle", "b"}:
                self.action_export_bundle()
                return
            if low in {"add", "add compare", "a"}:
                self.action_add_cmp()
                return
            if low in {"clear compare", "cc"}:
                self.action_clear_cmp()
                return
            if low in {"export compare", "mc", "m"}:
                self.action_export_cmp()
                return
            if low.startswith("sweep"):
                # Allow: sweep seeds=1,2,3 widths=8,16,32
                try:
                    parts = dict(p.split("=", 1) for p in s.split()[1:] if "=" in p)
                except Exception:
                    parts = {}
                if "seeds" in parts:
                    self.query_one("#sweep_seeds", Input).value = parts["seeds"]  # type: ignore[attr-defined]
                if "widths" in parts:
                    self.query_one("#sweep_widths", Input).value = parts["widths"]  # type: ignore[attr-defined]
                self.action_sweep()
                return
            if low.startswith("preset "):
                name = s.split(None, 1)[1].strip().lower()
                if name == "quick": self._on_preset_quick(); return
                if name == "ulam": self._on_preset_ulam(); return
                if name == "k": self._on_preset_k(); return
                if name == "health": self._on_preset_health(); return
                if name == "compare": self._on_preset_compare(); return
                if name == "monitor": self._on_preset_monitor(); return
                self._log(f"Unknown preset: {name}")
                return
            if low.startswith("provider "):
                prov = s.split(None, 1)[1]
                self.query_one("#provider", Input).value = prov  # type: ignore[attr-defined]
                self.action_apply_provider()
                return
            if low.startswith("set ") and "=" in s:
                body = s[4:]
                k, v = body.split("=", 1)
                self._cmd_set(k, v)
                return
            if low.startswith("toggle "):
                self._cmd_toggle(s.split(None, 1)[1])
                return
            if low in {"help", "h", "?"}:
                self.action_help()
                return
            self._log(f"Unknown command: {s}")

        from textual import on

        @on(Input.Submitted, "#cmdline")
        def _on_cmdline(self, ev: Input.Submitted) -> None:  # type: ignore
            self._cmd_exec(ev.value or "")
            ev.input.value = ""

    HelixProTUI().run()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run_pro_tui(argv)


if __name__ == "__main__":
    raise SystemExit(main())
