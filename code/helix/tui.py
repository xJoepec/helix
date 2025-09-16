from __future__ import annotations

from typing import Any, Dict, List, Optional

import importlib.util
import os
import time
from datetime import datetime

import numpy as np


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
        extract_partitions,
        build_V_from_incidence,
        sanity_check_ucp,
        ulam_pf,
        spectral_gap,
        region_counts,
        mass_consistency_errors,
    )
    from .serialize import to_jsonable
    from .cli import MLP, make_moons

    def step(msg: str):
        if progress_cb:
            progress_cb(msg)

    step("Generating dataset…")
    X, y = make_moons(n=int(params["samples"]), noise=float(params["noise"]), seed=int(params["seed"]))

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

    step("Extracting partitions…")
    af = extract_partitions(model, X)
    n_list = region_counts(af.B_list)
    errs = mass_consistency_errors(af.B_list, af.tau_list)

    step("Computing CP diagnostics…")
    cp_stats: List[Dict[str, float]] = []
    for k, B in enumerate(af.B_list, start=1):
        tau_prev = np.array([1.0]) if k == 1 else af.tau_list[k - 2]
        tau_cur = af.tau_list[k - 1]
        V = build_V_from_incidence(B, tau_prev, tau_cur)
        cp_stats.append(sanity_check_ucp(V, trials=6))

    step("Running Ulam PF…")
    lo = np.array([-2.0, -1.5])
    hi = np.array([3.0, 2.5])
    P, _ = ulam_pf(
        lambda z: z,  # identity as a placeholder map
        (lo, hi),
        bins_per_dim=int(params["ulam_bins"]),
        samples_per_cell=int(params["ulam_samples_per_cell"]),
    )
    gap = spectral_gap(P)

    step("Done.")
    metrics = {
        "region_counts": n_list,
        "mass_errors": errs,
        "cp_stats": cp_stats,
        "ulam_spectral_gap": gap,
    }
    return to_jsonable(metrics)


def run_tui(argv: Optional[List[str]] = None) -> int:
    if not _has_textual():
        print("Textual is not installed. Install Helix TUI extras: pip install '.[tui]'")
        return 1

    from textual.app import App, ComposeResult
    from textual.widgets import (
        Button,
        Footer,
        Header,
        Input,
        Label,
        Log,
        ProgressBar,
        Static,
        Switch,
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
    }

    class HelixTUI(App):
        CSS = """
        Screen { layout: vertical; }
        #controls { dock: top; padding: 1 2; }
        #buttons { padding-top: 1; }
        #logbox { height: 1fr; }
        """
        BINDINGS = [
            ("r", "run", "Run"),
            ("c", "cancel", "Cancel"),
            ("s", "save", "Save Config"),
            ("d", "reset", "Reset Defaults"),
            ("e", "export", "Export JSON"),
            ("q", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.params: Dict[str, Any] = _load_config(DEFAULTS)
            self._worker_running = False
            self._last_metrics: Optional[Dict[str, Any]] = None
            self._last_path: Optional[str] = None

        def compose(self) -> ComposeResult:  # type: ignore[override]
            yield Header()
            with Vertical(id="controls"):
                with Horizontal():
                    yield Input(str(self.params["samples"]), placeholder="samples", id="samples")
                    yield Input(str(self.params["noise"]), placeholder="noise", id="noise")
                    yield Input(str(self.params["seed"]), placeholder="seed", id="seed")
                    yield Input(str(self.params["width"]), placeholder="width", id="width")
                    yield Input(str(self.params["epochs"]), placeholder="epochs", id="epochs")
                with Horizontal():
                    yield Switch(value=bool(self.params["no_train"]), id="no_train", name="no_train")
                    yield Label("no_train")
                    yield Input(str(self.params["ulam_bins"]), placeholder="ulam_bins", id="ulam_bins")
                    yield Input(
                        str(self.params["ulam_samples_per_cell"]),
                        placeholder="ulam_samples_per_cell",
                        id="ulam_samples_per_cell",
                    )
                with Horizontal(id="buttons"):
                    yield Button("Run", id="run")
                    yield Button("Cancel", id="cancel", disabled=True)
                    yield Button("Save Config", id="save")
                    yield Button("Reset Defaults", id="reset")
                    yield Button("Export JSON", id="export", disabled=True)
            yield ProgressBar(total=6, id="progress")
            yield Log(id="logbox")
            yield Footer()

        def _read_params(self) -> None:
            def getv(id_: str, cast):
                w = self.query_one(f"#{id_}")
                if isinstance(w, Input):
                    return cast(w.value or w.placeholder)
                if isinstance(w, Switch):
                    return bool(w.value)
                return self.params[id_]

            self.params = {
                "samples": int(getv("samples", int)),
                "noise": float(getv("noise", float)),
                "seed": int(getv("seed", int)),
                "width": int(getv("width", int)),
                "epochs": int(getv("epochs", int)),
                "no_train": bool(getv("no_train", bool)),
                "ulam_bins": int(getv("ulam_bins", int)),
                "ulam_samples_per_cell": int(getv("ulam_samples_per_cell", int)),
            }

        def _set_running(self, running: bool) -> None:
            self._worker_running = running
            self.query_one("#run", Button).disabled = running
            self.query_one("#cancel", Button).disabled = not running
            # Export enabled only when we have metrics and not running
            self.query_one("#export", Button).disabled = running or (self._last_metrics is None)

        def _progress(self, msg: str) -> None:
            bar = self.query_one("#progress", ProgressBar)
            log = self.query_one("#logbox", Log)
            bar.advance(1)
            log.write_line(msg)
            self.refresh()

        def action_run(self) -> None:
            if self._worker_running:
                return
            self._read_params()
            self._last_metrics = None
            self._set_running(True)
            bar = self.query_one("#progress", ProgressBar)
            bar.reset(progress=0)
            log = self.query_one("#logbox", Log)
            log.clear()

            def work():
                try:
                    return _compute_metrics(self.params, progress_cb=lambda m: self.call_from_thread(self._progress, m))
                except Exception as e:  # pragma: no cover
                    self.call_from_thread(log.write_line, f"Error: {e}")
                    return None

            def done(result):
                self._set_running(False)
                if result is not None:
                    self._last_metrics = result
                    self.query_one("#export", Button).disabled = False

            # Run in a thread using Textual's worker API if available; else fallback
            try:
                worker = self.run_worker(work, thread=True, exclusive=True)
                worker.finished.connect(lambda res=None: done(worker.result))  # type: ignore[attr-defined]
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
            self.query_one("#logbox", Log).write_line("Cancelled.")

        def action_save(self) -> None:
            self._read_params()
            cfg = self.params.copy()
            cfg["version"] = "0.1"
            cfg["updated_at"] = datetime.utcnow().isoformat()
            _save_config(cfg)
            self.query_one("#logbox", Log).write_line("Config saved.")

        def action_reset(self) -> None:
            self.params = DEFAULTS.copy()
            self.query_one("#samples", Input).value = str(self.params["samples"])  # type: ignore[attr-defined]
            self.query_one("#noise", Input).value = str(self.params["noise"])  # type: ignore[attr-defined]
            self.query_one("#seed", Input).value = str(self.params["seed"])  # type: ignore[attr-defined]
            self.query_one("#width", Input).value = str(self.params["width"])  # type: ignore[attr-defined]
            self.query_one("#epochs", Input).value = str(self.params["epochs"])  # type: ignore[attr-defined]
            self.query_one("#no_train", Switch).value = bool(self.params["no_train"])  # type: ignore[attr-defined]
            self.query_one("#ulam_bins", Input).value = str(self.params["ulam_bins"])  # type: ignore[attr-defined]
            self.query_one("#ulam_samples_per_cell", Input).value = str(self.params["ulam_samples_per_cell"])  # type: ignore[attr-defined]
            self.query_one("#logbox", Log).write_line("Parameters reset to defaults.")

        def action_export(self) -> None:
            if not self._last_metrics:
                return
            from .serialize import json_dumps

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fn = f"helix_metrics_{ts}.json"
            with open(fn, "w", encoding="utf-8") as f:
                f.write(json_dumps({
                    "config": self.params,
                    "metrics": self._last_metrics,
                }, indent=2))
            self._last_path = os.path.abspath(fn)
            self.query_one("#logbox", Log).write_line(f"Exported to {self._last_path}")

    HelixTUI().run()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return run_tui(argv)


if __name__ == "__main__":
    raise SystemExit(main())
