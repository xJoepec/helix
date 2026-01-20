from __future__ import annotations

import importlib.util
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# Singleton storage for the HelixTUI class so we can expose it lazily while
# keeping Textual as an optional dependency.
HelixTUI: Optional[Any] = None
_HELIX_TUI_BUILD_ERROR: Optional[Exception] = None


def _has_textual() -> bool:
    return importlib.util.find_spec("textual") is not None


def _user_config_path() -> str:
    try:
        import platformdirs  # type: ignore

        cfg_dir = platformdirs.user_config_dir("helix", "helix")
    except Exception:
        cfg_dir = os.path.join(os.path.expanduser("~"), ".config", "helix")
    os.makedirs(cfg_dir, exist_ok=True)
    return os.path.join(cfg_dir, "config.toml")


def _save_config(cfg: Dict[str, Any]) -> None:
    try:
        import toml  # type: ignore

        path = _user_config_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            toml.dump(cfg, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_config(defaults: Dict[str, Any]) -> Dict[str, Any]:
    try:
        import toml  # type: ignore

        path = _user_config_path()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = toml.load(f)
            out = defaults.copy()
            out.update({k: cfg.get(k, v) for k, v in defaults.items()})
            return out
    except Exception:
        pass
    return defaults.copy()


def _compute_metrics(params: Dict[str, Any], progress_cb=None) -> Dict[str, Any]:
    # Lazy imports to avoid heavy deps at import-time
    from . import (
        build_V_from_incidence,
        extract_partitions,
        mass_consistency_errors,
        region_counts,
        sanity_check_ucp,
        spectral_gap,
        ulam_pf,
    )
    from .cli import MLP, make_moons
    from .ktheory import k_invariants_from_B as _k_inv  # optional usage
    from .serialize import to_jsonable

    def step(msg: str):
        if progress_cb:
            progress_cb(msg)

    # Read feature toggles from params with safe defaults to preserve
    # backward compatibility when called from helixenv.
    do_regions = bool(params.get("regions", True))
    do_mass = bool(params.get("mass", True))
    do_cp = bool(params.get("cp", True))
    do_ulam = bool(params.get("ulam", True))
    do_sparse = bool(params.get("sparse", True))
    do_ktheory = bool(params.get("ktheory", False))

    step("Generating dataset…")
    X, y = make_moons(
        n=int(params["samples"]), noise=float(params["noise"]), seed=int(params["seed"])
    )

    step("Building model…")
    width = int(params["width"])
    model = MLP(d_in=2, widths=(width, width), d_out=2)

    if not bool(params.get("no_train", False)):
        step("Training model…")
        import torch
        import torch.nn.functional as F

        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        for _ in range(int(params["epochs"])):
            opt.zero_grad()
            logits = model(X_t)
            loss = F.cross_entropy(logits, y_t)
            loss.backward()
            opt.step()

    af = None
    n_list: Optional[List[int]] = None
    errs: Optional[List[float]] = None
    if do_regions or do_mass or do_cp or do_sparse or do_ktheory:
        step("Extracting partitions…")
        af = extract_partitions(model, X)
        if do_regions:
            n_list = region_counts(af.B_list)
        if do_mass:
            errs = mass_consistency_errors(af.B_list, af.tau_list)

    cp_stats: Optional[List[Dict[str, float]]] = None
    if do_cp and af is not None:
        step("Computing CP diagnostics…")
        cp_stats = []
        for k, B in enumerate(af.B_list, start=1):
            tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
            tau_cur = af.tau_list[k - 1]
            V = build_V_from_incidence(B, tau_prev, tau_cur)
            cp_stats.append(sanity_check_ucp(V, trials=6))

    gap: Optional[float] = None
    if do_ulam:
        step("Running Ulam PF…")
        lo = np.array([-2.0, -1.5])
        hi = np.array([3.0, 2.5])
        # Use a small residual block map (Use Case 4)
        try:
            import torch
            import torch.nn as nn
        except Exception:  # pragma: no cover
            torch = None  # type: ignore
            nn = None  # type: ignore

        if torch is not None and nn is not None:
            h2 = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2))
            with torch.no_grad():
                for p in h2.parameters():
                    p.mul_(0.1)

            def F_block(x_np: np.ndarray, eps: float = 0.3) -> np.ndarray:
                x_t = torch.from_numpy(x_np.astype(np.float32))
                y_ = x_t + eps * h2(x_t)
                return y_.detach().numpy().astype(np.float64)

            eps = float(params.get("ulam_eps", 0.4))
            P, _ = ulam_pf(
                lambda z: F_block(z, eps=eps),
                (lo, hi),
                bins_per_dim=int(params["ulam_bins"]),
                samples_per_cell=int(params["ulam_samples_per_cell"]),
            )
        else:
            # Fallback: identity map to keep UI functional without torch
            P, _ = ulam_pf(
                lambda z: z,
                (lo, hi),
                bins_per_dim=int(params["ulam_bins"]),
                samples_per_cell=int(params["ulam_samples_per_cell"]),
            )
        gap = spectral_gap(P)

    sparse_stats: Optional[List[Dict[str, Any]]] = None
    if do_sparse and af is not None:
        step("Summarizing sparse structure…")
        sparse_stats = []
        for k, B in enumerate(af.B_list, start=1):
            n_prev, n_cur = B.shape
            nnz = int(B.sum())
            parent_ptrs = int(len(af.parent_of_list[k - 1]))
            sparse_stats.append(
                {
                    "depth": k,
                    "shape": [n_prev, n_cur],
                    "nnz": nnz,
                    "parent_ptrs": parent_ptrs,
                    "sparsity": float(1.0 - (nnz / max(1, n_prev * n_cur))),
                }
            )

    k_list: Optional[List[Dict[str, Any]]] = None
    if do_ktheory and af is not None:
        step("Computing K-theory invariants…")
        k_list = []
        for k, B in enumerate(af.B_list, start=1):
            try:
                inv = _k_inv(B)
                k_list.append(
                    {
                        "depth": k,
                        "rank": int(inv.get("rank", 0)),
                        "nullity": int(inv.get("nullity", 0)),
                        "torsion": [int(t) for t in inv.get("torsion", [])],
                        "S_diag": [int(d) for d in inv.get("S_diag", [])],
                    }
                )
            except Exception as e:
                k_list.append({"depth": k, "error": str(e)})

    step("Done.")
    metrics: Dict[str, Any] = {}
    if n_list is not None:
        metrics["region_counts"] = n_list
    if errs is not None:
        metrics["mass_errors"] = errs
    if cp_stats is not None:
        metrics["cp_stats"] = cp_stats
    if gap is not None:
        metrics["ulam_spectral_gap"] = gap
    if sparse_stats is not None:
        metrics["sparse_stats"] = sparse_stats
    if k_list is not None:
        metrics["k_invariants"] = k_list
    return to_jsonable(metrics)


def _ensure_tui_class():
    """Builds (and caches) the HelixTUI class once Textual is available."""

    global HelixTUI

    if HelixTUI is not None:
        return HelixTUI

    if not _has_textual():
        raise ImportError("Textual is not installed. Install Helix TUI extras: pip install '.[tui]'")

    from textual import on
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, VerticalScroll
    from textual.widgets import (
        Checkbox,
        Footer,
        Input,
        Label,
        ProgressBar,
        Static,
        Switch,
    )
    from textual.worker import Worker

    # Import our custom themed widgets and theme
    try:
        from .tui_theme import generate_css
        from .tui_widgets import (
            ButtonGroup,
            Divider,
            DNAHelixHeader,
            RichLog,
            SectionHeader,
            StyledText,
            ThemedButton,
            ThemedDataTable,
        )
    except ImportError:
        # Fallback for direct imports
        from tui_theme import generate_css
        from tui_widgets import (
            ButtonGroup,
            Divider,
            DNAHelixHeader,
            RichLog,
            SectionHeader,
            StyledText,
            ThemedButton,
            ThemedDataTable,
        )

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
        # Feature toggles mapping to uses.md
        "regions": True,
        "mass": True,
        "cp": True,
        "ulam": True,
        "sparse": True,
        "ktheory": False,
    }

    class HelixTUI(App):
        # Use our comprehensive theme CSS
        CSS = generate_css() + """
        /* Additional TUI-specific styles */
        Screen {
            layout: vertical;
            background: $background;
        }

        .hidden {
            display: none;
        }

        #controls-container {
            dock: top;
            padding: 1 2;
            background: $surface;
            border-bottom: solid $border;
        }

        .input-group {
            height: auto;
            margin: 1 0;
        }

        .input-group Input {
            width: 1fr;
            margin: 0 1;
        }

        .input-group Label {
            width: auto;
            margin: 0 1;
            color: $subtle;
        }

        #buttons {
            margin-top: 1;
        }

        #tables-container {
            height: 1fr;
            padding: 1 2;
        }

        #help {
            height: 12;
            display: none;
        }

        #help.visible {
            display: block;
        }

        #cmdbar {
            height: 3;
            padding: 0 1;
            background: $surface;
            border-top: solid $border;
        }

        #logbox {
            height: 1fr;
            scrollbar-size: 1 1;
        }

        .checkbox-group {
            margin: 1 0;
        }

        .checkbox-group Checkbox {
            margin: 0 1;
        }
        """
        BINDINGS = [
            ("r", "run", "Run"),
            ("c", "cancel", "Cancel"),
            ("s", "save", "Save Config"),
            ("d", "reset", "Reset Defaults"),
            ("e", "export", "Export JSON"),
            ("x", "export_csv", "Export CSVs"),
            (":", "command", "Command"),
            ("h", "help", "Help"),
            ("q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.params: Dict[str, Any] = _load_config(DEFAULTS)
            self._worker_running = False
            self._last_metrics: Optional[Dict[str, Any]] = None
            self._last_path: Optional[str] = None
            self._loading = True  # Start in loading state, show only animation
            # Local toggles separate from params for clarity in UI
            self.which: Dict[str, bool] = {
                "regions": bool(self.params.get("regions", True)),
                "mass": bool(self.params.get("mass", True)),
                "cp": bool(self.params.get("cp", True)),
                "ulam": bool(self.params.get("ulam", True)),
                "sparse": bool(self.params.get("sparse", True)),
                "ktheory": bool(self.params.get("ktheory", False)),
            }

        def compose(self) -> ComposeResult:  # type: ignore[override]
            # Dock the header at the top to prevent layout issues
            header = DNAHelixHeader()
            header.styles.dock = "top"
            yield header

            # Main layout - initially hidden if loading
            with Horizontal(id="main-layout", classes="hidden" if self._loading else ""):
                with Vertical(id="left-pane", classes="panel"):
                    with VerticalScroll(id="controls-scroll"):
                        yield SectionHeader("Dataset Configuration")
                        with Vertical(classes="input-group"):
                            with Horizontal():
                                yield Input(
                                    str(self.params["samples"]),
                                    placeholder="samples",
                                    id="samples"
                                )
                                yield Input(
                                    str(self.params["noise"]),
                                    placeholder="noise",
                                    id="noise"
                                )
                                yield Input(
                                    str(self.params["seed"]),
                                    placeholder="seed",
                                    id="seed"
                                )

                        yield SectionHeader("Model Architecture")
                        with Vertical(classes="input-group"):
                            with Horizontal():
                                yield Input(
                                    str(self.params["width"]),
                                    placeholder="width",
                                    id="width"
                                )
                                yield Input(
                                    str(self.params["epochs"]),
                                    placeholder="epochs",
                                    id="epochs"
                                )
                                yield Switch(
                                    value=bool(self.params["no_train"]),
                                    id="no_train",
                                    name="no_train"
                                )
                                yield Label("Skip Training")

                        yield SectionHeader("Analysis Parameters")
                        with Vertical(classes="input-group"):
                            with Horizontal():
                                yield Input(
                                    str(self.params["ulam_bins"]), placeholder="ulam_bins", id="ulam_bins"
                                )
                                yield Input(
                                    str(self.params["ulam_samples_per_cell"]),
                                    placeholder="ulam_samples_per_cell",
                                    id="ulam_samples_per_cell",
                                )
                                yield Input(str(self.params["ulam_eps"]), placeholder="ulam_eps", id="ulam_eps")

                        yield SectionHeader("Analysis Features")
                        with Horizontal(classes="checkbox-group"):
                            yield Checkbox("Region growth", value=self.which["regions"], id="regions")
                            yield Checkbox("Mass consistency", value=self.which["mass"], id="mass")
                            yield Checkbox("CP checks", value=self.which["cp"], id="cp")
                            yield Checkbox("Ulam mixing", value=self.which["ulam"], id="ulam")
                            yield Checkbox("Sparse stats", value=self.which["sparse"], id="sparse")
                            yield Checkbox("K-theory", value=self.which["ktheory"], id="ktheory")

                        yield Divider()
                        with Horizontal(id="buttons"):
                            with ButtonGroup(group_type="actions"):
                                yield ThemedButton("▶ Run", button_type="primary", id="run")
                                yield ThemedButton("■ Cancel", button_type="danger", id="cancel", disabled=True)
                            with ButtonGroup(group_type="export"):
                                yield ThemedButton("📄 Export JSON", button_type="secondary", id="export", disabled=True)
                                yield ThemedButton("📊 Export CSV", button_type="secondary", id="export_csv", disabled=True)
                            with ButtonGroup(group_type="config"):
                                yield ThemedButton("💾 Save", id="save")
                                yield ThemedButton("↺ Reset", id="reset")

                    yield Divider()
                    yield StyledText("RUN MONITOR", style_type="headline")
                    yield ProgressBar(total=8, id="progress")

                    yield Divider()
                    yield StyledText("COMMAND PALETTE", style_type="subtle")
                    with Horizontal(id="cmdbar", classes="command-input"):
                        yield StyledText(":", style_type="accent")
                        yield Input(
                            placeholder="run | set samples=2000 | toggle cp | export csv | help",
                            id="cmdline",
                        )
                    yield StyledText("CONSOLE FEED", style_type="subtle")
                    yield RichLog(id="logbox")

                with Vertical(id="right-pane", classes="panel"):
                    yield StyledText("ANALYTICS HUD", style_type="headline")
                    with VerticalScroll(id="tables-container"):
                        yield Divider()
                        yield ThemedDataTable(id="t_regions")
                        yield ThemedDataTable(id="t_mass")
                        yield ThemedDataTable(id="t_cp")
                        yield ThemedDataTable(id="t_ulam")
                        yield ThemedDataTable(id="t_sparse")
                        yield ThemedDataTable(id="t_k")
                    yield Divider()
                    yield StyledText("Help (press 'h' to toggle)", style_type="subtle")
                    with VerticalScroll(id="help", classes="help-panel"):
                        yield Static("Loading help…", id="help_text")

            yield Footer()

        def on_mount(self) -> None:
            # Show main content after a brief delay (for animation to be visible)
            if self._loading:
                self.set_timer(0.5, self._show_main_content)

            try:
                self.query_one("#controls-scroll").scroll_home(animate=False)
                self.query_one("#tables-container").scroll_home(animate=False)
            except Exception:
                pass

        def _show_main_content(self) -> None:
            """Show the main content after initial animation display."""
            self._loading = False
            main_layout = self.query_one("#main-layout")
            main_layout.remove_class("hidden")

        def _read_params(self) -> None:
            def getv(id_: str, cast):
                w = self.query_one(f"#{id_}")
                if isinstance(w, Input):
                    return cast(w.value or w.placeholder)
                if isinstance(w, Switch):
                    return bool(w.value)
                return self.params[id_]

            # Sync params and toggles from UI
            self.params = {
                "samples": int(getv("samples", int)),
                "noise": float(getv("noise", float)),
                "seed": int(getv("seed", int)),
                "width": int(getv("width", int)),
                "epochs": int(getv("epochs", int)),
                "no_train": bool(getv("no_train", bool)),
                "ulam_bins": int(getv("ulam_bins", int)),
                "ulam_samples_per_cell": int(getv("ulam_samples_per_cell", int)),
                "ulam_eps": float(getv("ulam_eps", float)),
                # toggles also stored in params for persistence/exports
                "regions": bool(self.query_one("#regions", Checkbox).value),
                "mass": bool(self.query_one("#mass", Checkbox).value),
                "cp": bool(self.query_one("#cp", Checkbox).value),
                "ulam": bool(self.query_one("#ulam", Checkbox).value),
                "sparse": bool(self.query_one("#sparse", Checkbox).value),
                "ktheory": bool(self.query_one("#ktheory", Checkbox).value),
            }
            # Mirror into self.which for easy access
            for k in self.which.keys():
                self.which[k] = bool(self.params.get(k, False))

        def _set_running(self, running: bool) -> None:
            self._worker_running = running
            self.query_one("#run", ThemedButton).disabled = running
            self.query_one("#cancel", ThemedButton).disabled = not running
            # Export enabled only when we have metrics and not running
            self.query_one("#export", ThemedButton).disabled = running or (self._last_metrics is None)
            self.query_one("#export_csv", ThemedButton).disabled = running or (self._last_metrics is None)

        def _progress(self, msg: str) -> None:
            bar = self.query_one("#progress", ProgressBar)
            log = self.query_one("#logbox", RichLog)
            bar.advance(1)
            log.write_info(msg)
            self.refresh()

        def action_run(self) -> None:
            if self._worker_running:
                return
            self._read_params()
            self._last_metrics = None
            self._set_running(True)
            bar = self.query_one("#progress", ProgressBar)
            try:
                bar.reset(progress=0)
            except Exception:
                try:
                    bar.progress = 0  # type: ignore[attr-defined]
                except Exception:
                    pass
            log = self.query_one("#logbox", RichLog)
            log.clear()

            def work():
                try:
                    return _compute_metrics(
                        self.params, progress_cb=lambda m: self.call_from_thread(self._progress, m)
                    )
                except Exception as e:  # pragma: no cover
                    self.call_from_thread(log.write_error, f"Error: {e}")
                    return None

            def done(result):
                self._set_running(False)
                if result is not None:
                    self._last_metrics = result
                    self.query_one("#export", ThemedButton).disabled = False
                    self.query_one("#export_csv", ThemedButton).disabled = False
                    # Populate tables for quick visual scan
                    self._populate_tables(result)
                    log.write_success("Analysis complete!")
                    self.refresh()

            # Store done callback for worker state handler
            self._analysis_done_callback = done

            # Run in a thread using Textual's worker API if available; else fallback
            try:
                self._analysis_worker = self.run_worker(work, thread=True, exclusive=True)
                # Worker completion will be handled by on_worker_state_changed
            except Exception:
                # Fallback: blocking (not ideal) but maintains functionality
                res = work()
                done(res)

        def action_cancel(self) -> None:
            # Best-effort: Textual worker cancellation if available
            try:
                from textual.worker import get_current_worker  # type: ignore

                w = get_current_worker()
                if w is not None:
                    w.cancel()
            except Exception:
                pass
            self._set_running(False)
            self.query_one("#logbox", RichLog).write_warning("Analysis cancelled by user.")

        def on_worker_state_changed(self, event: "Worker.StateChanged") -> None:
            """Handle worker completion."""
            from textual.worker import WorkerState

            # Check if this is our analysis worker
            if hasattr(self, '_analysis_worker') and event.worker is self._analysis_worker:
                if event.state == WorkerState.SUCCESS:
                    # Call the done callback with the worker result
                    if hasattr(self, '_analysis_done_callback'):
                        self._analysis_done_callback(event.worker.result)
                elif event.state == WorkerState.ERROR:
                    # Handle error
                    self._set_running(False)
                    self.query_one("#logbox", RichLog).write_error(f"Analysis failed: {event.worker.error}")
                elif event.state == WorkerState.CANCELLED:
                    # Already handled in action_cancel
                    pass

        def action_save(self) -> None:
            self._read_params()
            cfg = self.params.copy()
            cfg["version"] = "0.1"
            cfg["updated_at"] = datetime.utcnow().isoformat()
            _save_config(cfg)
            self.query_one("#logbox", RichLog).write_success("Configuration saved successfully.")

        def action_reset(self) -> None:
            self.params = DEFAULTS.copy()
            self.query_one("#samples", Input).value = str(self.params["samples"])  # type: ignore[attr-defined]
            self.query_one("#noise", Input).value = str(self.params["noise"])  # type: ignore[attr-defined]
            self.query_one("#seed", Input).value = str(self.params["seed"])  # type: ignore[attr-defined]
            self.query_one("#width", Input).value = str(self.params["width"])  # type: ignore[attr-defined]
            self.query_one("#epochs", Input).value = str(self.params["epochs"])  # type: ignore[attr-defined]
            self.query_one("#no_train", Switch).value = bool(self.params["no_train"])  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(
                self.params["ulam_samples_per_cell"]
            )  # type: ignore[attr-defined]
            self.query_one("#ulam_eps", Input).value = str(self.params["ulam_eps"])  # type: ignore[attr-defined]
            for k in ("regions", "mass", "cp", "ulam", "sparse", "ktheory"):
                self.query_one(f"#{k}", Checkbox).value = bool(self.params[k])  # type: ignore[attr-defined]
            self.query_one("#logbox", RichLog).write_info("Parameters reset to defaults.")

        def action_export(self) -> None:
            if not self._last_metrics:
                return
            from .serialize import json_dumps

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"helix_metrics_{ts}.json"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(
                    json_dumps(
                        {
                            "config": self.params,
                            "metrics": self._last_metrics,
                        },
                        indent=2,
                    )
                )
            self._last_path = os.path.abspath(fn)
            self.query_one("#logbox", RichLog).write_success(f"📄 Exported to {self._last_path}")

        def action_help(self) -> None:
            # Toggle help panel and load docs
            try:
                cont = self.query_one("#help")
                cont.display = not getattr(cont, "display", True)
            except Exception:
                pass
            try:
                txt = self.query_one("#help_text", Static)
                root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
                )
                paths = [
                    os.path.join(root, "docs", "applications.md"),
                    os.path.join(root, "uses.md"),
                ]
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
            # Accept both params and toggles
            k = key.strip().lower()
            v = val.strip()
            try:
                if k in {
                    "samples",
                    "seed",
                    "width",
                    "epochs",
                    "ulam_bins",
                    "ulam_samples_per_cell",
                }:
                    self.query_one(f"#{k}", Input).value = str(int(float(v)))  # type: ignore[attr-defined]
                elif k in {"noise", "ulam_eps"}:
                    self.query_one(f"#{k}", Input).value = str(float(v))  # type: ignore[attr-defined]
                elif k in {"no_train", "regions", "mass", "cp", "ulam", "sparse", "ktheory"}:
                    from textual.widgets import Checkbox as _CB

                    self.query_one(f"#{k}", _CB).value = v.lower() in {
                        "1",
                        "true",
                        "yes",
                        "y",
                        "on",
                    }  # type: ignore[attr-defined]
                else:
                    self.query_one("#logbox", RichLog).write_error(f"Unknown key: {k}")
                    return
                self.query_one("#logbox", RichLog).write_info(f"Set {k}={v}")
            except Exception as e:
                self.query_one("#logbox", RichLog).write_error(f"Set error: {e}")

        def _cmd_toggle(self, key: str) -> None:
            from textual.widgets import Checkbox as _CB

            try:
                cb = self.query_one(f"#{key}", _CB)
                cb.value = not bool(cb.value)
                self.query_one("#logbox", RichLog).write_info(f"Toggled {key} -> {cb.value}")
            except Exception:
                self.query_one("#logbox", RichLog).write_error(f"Unknown toggle: {key}")

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
            if low in {"reset", "d"}:
                self.action_reset()
                return
            if low in {"save", "s"}:
                self.action_save()
                return
            if low in {"help", "h", "?"}:
                self.action_help()
                return
            if low.startswith("set ") and "=" in s:
                body = s[4:]
                k, v = body.split("=", 1)
                self._cmd_set(k, v)
                return
            if low.startswith("toggle "):
                self._cmd_toggle(s.split(None, 1)[1])
                return
            self.query_one("#logbox", RichLog).write_error(f"Unknown command: {s}")

        @on(Input.Submitted, "#cmdline")
        def _on_cmdline(self, ev: Input.Submitted) -> None:  # type: ignore
            self._cmd_exec(ev.value or "")
            ev.input.value = ""

        def action_export_csv(self) -> None:
            if not self._last_metrics:
                return
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"helix_metrics_{ts}_"
            out_files: List[str] = []

            def w(name: str, header: List[str], rows: List[List[Any]]):
                path = base + name
                with open(path, "w", encoding="utf-8") as f:
                    f.write(",".join(header) + "\n")
                    for r in rows:
                        f.write(",".join(str(x) for x in r) + "\n")
                out_files.append(os.path.abspath(path))

            m = self._last_metrics
            # Regions
            if "region_counts" in m:
                rows = [[i + 1, v] for i, v in enumerate(m["region_counts"])]
                w("regions.csv", ["depth", "regions"], rows)
            # Mass
            if "mass_errors" in m:
                rows = [[i + 1, v] for i, v in enumerate(m["mass_errors"])]
                w("mass.csv", ["depth", "l1_error"], rows)
            # CP stats
            if "cp_stats" in m:
                rows = []
                for i, s in enumerate(m["cp_stats"], start=1):
                    rows.append(
                        [
                            i,
                            s.get("unital_err_fro", 0.0),
                            s.get("coisometry_err_fro", 0.0),
                            s.get("psd_min_eig_violation", 0.0),
                        ]
                    )
                w("cp.csv", ["depth", "unital", "coiso", "psd_vio"], rows)
            # Ulam
            if "ulam_spectral_gap" in m:
                w(
                    "ulam.csv",
                    ["metric", "value"],
                    [["spectral_gap", f"{float(m['ulam_spectral_gap']):.4f}"]],
                )
            # Sparse
            if "sparse_stats" in m:
                rows = []
                for d in m["sparse_stats"]:
                    n_prev, n_cur = d.get("shape", [0, 0])
                    rows.append(
                        [
                            d.get("depth", 0),
                            n_prev,
                            n_cur,
                            d.get("nnz", 0),
                            d.get("parent_ptrs", 0),
                            f"{float(d.get('sparsity', 0.0)):.4%}",
                        ]
                    )
                w(
                    "sparse.csv",
                    ["depth", "n_prev", "n_cur", "nnz", "parent_ptrs", "sparsity"],
                    rows,
                )
            # K-theory
            if "k_invariants" in m:
                rows = []
                for ki in m["k_invariants"]:
                    if "error" in ki:
                        rows.append([ki.get("depth", 0), "error", ki.get("error", "")])
                    else:
                        rows.append(
                            [
                                ki.get("depth", 0),
                                ki.get("rank", 0),
                                ki.get("nullity", 0),
                                ";".join(map(str, ki.get("torsion", []))),
                                ";".join(map(str, ki.get("S_diag", []))),
                            ]
                        )
                w("ktheory.csv", ["depth", "rank", "nullity", "torsion", "S_diag"], rows)
            for p in out_files:
                self.query_one("#logbox", RichLog).write_success(f"📊 Exported {p}")

        def _populate_tables(self, res: Dict[str, Any]) -> None:
            def setup(dt: ThemedDataTable, cols: List[str], rows: List[List[Any]],
                     col_styles: Optional[List[Optional[str]]] = None) -> None:
                dt.clear(columns=True)

                # Add columns with optional styling
                if col_styles:
                    for c, style in zip(cols, col_styles):
                        dt.add_column_with_style(c, style_type=style)
                else:
                    for c in cols:
                        dt.add_column(c)

                for r in rows:
                    dt.add_row(*[str(x) for x in r])

            # Regions
            dt = self.query_one("#t_regions", ThemedDataTable)
            if "region_counts" in res:
                setup(
                    dt,
                    ["depth", "regions"],
                    [[i + 1, v] for i, v in enumerate(res["region_counts"])],
                )
                dt.display = True
            else:
                dt.display = False

            # Mass
            dt = self.query_one("#t_mass", ThemedDataTable)
            if "mass_errors" in res:
                setup(
                    dt,
                    ["depth", "l1_error"],
                    [[i + 1, v] for i, v in enumerate(res["mass_errors"])],
                )
                dt.display = True
            else:
                dt.display = False

            # CP
            dt = self.query_one("#t_cp", ThemedDataTable)
            if "cp_stats" in res:
                rows = []
                for i, s in enumerate(res["cp_stats"], start=1):
                    rows.append(
                        [
                            i,
                            s.get("unital_err_fro", 0.0),
                            s.get("coisometry_err_fro", 0.0),
                            s.get("psd_min_eig_violation", 0.0),
                        ]
                    )
                setup(dt, ["depth", "unital", "coiso", "psd_vio"], rows)
                dt.display = True
            else:
                dt.display = False

            # Ulam
            dt = self.query_one("#t_ulam", ThemedDataTable)
            if "ulam_spectral_gap" in res:
                setup(
                    dt,
                    ["metric", "value"],
                    [["spectral_gap", f"{float(res['ulam_spectral_gap']):.4f}"]],
                )
                dt.display = True
            else:
                dt.display = False

            # Sparse
            dt = self.query_one("#t_sparse", ThemedDataTable)
            if "sparse_stats" in res:
                rows = []
                for d in res["sparse_stats"]:
                    n_prev, n_cur = d.get("shape", [0, 0])
                    rows.append(
                        [
                            d.get("depth", 0),
                            n_prev,
                            n_cur,
                            d.get("nnz", 0),
                            d.get("parent_ptrs", 0),
                            f"{float(d.get('sparsity', 0.0)):.2%}",
                        ]
                    )
                setup(dt, ["depth", "n_prev", "n_cur", "nnz", "parent_ptrs", "sparsity"], rows)
                dt.display = True
            else:
                dt.display = False

            # K-theory
            dt = self.query_one("#t_k", ThemedDataTable)
            if "k_invariants" in res:
                rows = []
                for ki in res["k_invariants"]:
                    if "error" in ki:
                        rows.append([ki.get("depth", 0), "error", ki.get("error", "")])
                    else:
                        rows.append(
                            [
                                ki.get("depth", 0),
                                ki.get("rank", 0),
                                ki.get("nullity", 0),
                                ",".join(map(str, ki.get("torsion", []))),
                                ",".join(map(str, ki.get("S_diag", []))),
                            ]
                        )
                setup(dt, ["depth", "rank", "nullity", "torsion", "S_diag"], rows)
                dt.display = True
            else:
                dt.display = False

    return HelixTUI


try:
    HelixTUI = _ensure_tui_class()
    _HELIX_TUI_BUILD_ERROR = None
except Exception as exc:  # pragma: no cover - construction happens lazily in tests
    HelixTUI = None
    _HELIX_TUI_BUILD_ERROR = exc


def run_tui(argv: Optional[List[str]] = None) -> int:
    """Entry point used by helix tui commands."""

    del argv  # Currently unused but kept for CLI compatibility.

    global HelixTUI, _HELIX_TUI_BUILD_ERROR

    try:
        cls = _ensure_tui_class()
        _HELIX_TUI_BUILD_ERROR = None
    except Exception as exc:  # pragma: no cover - depends on optional deps
        cls = None
        _HELIX_TUI_BUILD_ERROR = exc

    if cls is None:
        if isinstance(_HELIX_TUI_BUILD_ERROR, ImportError):
            print("Textual is not installed. Install Helix TUI extras: pip install '.[tui]'")
        else:
            detail = "Helix TUI not available."
            if _HELIX_TUI_BUILD_ERROR is not None:
                detail = f"{detail} Details: {_HELIX_TUI_BUILD_ERROR}"
            print(detail)
        return 1

    HelixTUI = cls
    cls().run()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run_tui(argv)


if __name__ == "__main__":
    raise SystemExit(main())
