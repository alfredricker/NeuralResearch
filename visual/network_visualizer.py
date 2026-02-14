from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from src.region.region import BaseRegion, EffectorRegion, RelayRegion, SensoryLevelRegion


StepFn = Callable[[int], None]


REGION_COLORS = {
    "sensory": "#8ecae6",
    "relay": "#cdb4db",
    "effector": "#ffafcc",
    "unknown": "#adb5bd",
}

NEURON_COLORS = {
    "SensoryNeuron": "#8ecae6",
    "StandardNeuron": "#cdb4db",
    "EffectorNeuron": "#ffafcc",
    "BaseNeuron": "#adb5bd",
}


def _safe_activity(value: float) -> str:
    return f"{value:.3f}"


@dataclass
class RegionLayout:
    region_id: str
    kind: str
    x: float
    y: float
    radius: float = 30.0


class NetworkVisualizer:
    """
    Interactive visualizer for region graph + selected region internals.

    Expected network object shape:
    - network.regions: Dict[str, BaseRegion]
    - network.edges: iterable of objects with src_region_id, dst_region_id, pattern, weight
    """

    def __init__(
        self,
        network: Any,
        step_fn: Optional[StepFn] = None,
        title: str = "Cortical Network Visualizer",
    ):
        self.network = network
        self.step_fn = step_fn if step_fn is not None else self._default_step
        self.title = title

        self.time_step = 0
        self.selected_region_id: Optional[str] = None

        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1500x900")

        self.header_var = tk.StringVar(value=self._header_text())
        self.status_var = tk.StringVar(value="Click a region to inspect neurons.")

        self._region_item_to_id: Dict[int, str] = {}
        self._region_layouts: Dict[str, RegionLayout] = {}

        self._build_ui()
        self._draw_region_graph()

    def _default_step(self, ticks: int) -> None:
        # Fallback when only a network is provided.
        if not hasattr(self.network, "step"):
            raise ValueError("No step function provided and network has no step() method")
        for _ in range(ticks):
            self.network.step(include_feed_in=True)

    def _header_text(self) -> str:
        return f"Time step: {self.time_step}"

    def _build_ui(self) -> None:
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top, textvariable=self.header_var, font=("Arial", 14, "bold")).pack(
            side=tk.LEFT, padx=10, pady=8
        )
        tk.Button(top, text="Step +1", command=lambda: self._advance(1), width=12).pack(
            side=tk.LEFT, padx=6
        )
        tk.Button(top, text="Step +10", command=lambda: self._advance(10), width=12).pack(
            side=tk.LEFT, padx=6
        )
        tk.Label(top, textvariable=self.status_var).pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.region_canvas = tk.Canvas(body, bg="#f8f9fa", width=760, height=800)
        self.region_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.region_canvas.bind("<Button-1>", self._on_region_click)

        self.detail_canvas = tk.Canvas(body, bg="#ffffff", width=740, height=800)
        self.detail_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _region_kind(self, region: BaseRegion) -> str:
        if isinstance(region, SensoryLevelRegion):
            return "sensory"
        if isinstance(region, EffectorRegion):
            return "effector"
        if isinstance(region, RelayRegion):
            return "relay"
        return "unknown"

    def _grouped_region_ids(self) -> Dict[str, List[str]]:
        groups: Dict[str, List[str]] = {"sensory": [], "relay": [], "effector": [], "unknown": []}
        for region_id, region in self.network.regions.items():
            groups[self._region_kind(region)].append(region_id)
        for ids in groups.values():
            ids.sort()
        return groups

    def _draw_region_graph(self) -> None:
        self.region_canvas.delete("all")
        self._region_item_to_id.clear()
        self._region_layouts.clear()

        w = max(self.region_canvas.winfo_width(), 760)
        h = max(self.region_canvas.winfo_height(), 800)

        groups = self._grouped_region_ids()
        x_positions = {
            "sensory": w * 0.20,
            "relay": w * 0.50,
            "effector": w * 0.80,
            "unknown": w * 0.50,
        }

        for kind, region_ids in groups.items():
            if not region_ids:
                continue
            gap = h / (len(region_ids) + 1)
            for i, region_id in enumerate(region_ids, start=1):
                self._region_layouts[region_id] = RegionLayout(
                    region_id=region_id,
                    kind=kind,
                    x=x_positions[kind],
                    y=i * gap,
                )

        # Draw region edges first (under nodes).
        for edge in getattr(self.network, "edges", []):
            src = self._region_layouts.get(edge.src_region_id)
            dst = self._region_layouts.get(edge.dst_region_id)
            if src is None or dst is None:
                continue

            self.region_canvas.create_line(
                src.x,
                src.y,
                dst.x,
                dst.y,
                arrow=tk.LAST,
                width=2,
                fill="#495057",
            )
            mx = (src.x + dst.x) / 2
            my = (src.y + dst.y) / 2
            pattern_label = getattr(edge.pattern, "kind", edge.pattern)
            pattern_label = getattr(pattern_label, "value", str(pattern_label))
            self.region_canvas.create_text(
                mx,
                my - 10,
                text=f"{pattern_label} w={edge.weight:.2f}",
                fill="#212529",
                font=("Arial", 9),
            )

        # Draw region nodes.
        for region_id, layout in self._region_layouts.items():
            region = self.network.regions[region_id]
            color = REGION_COLORS.get(layout.kind, REGION_COLORS["unknown"])
            r = layout.radius
            item = self.region_canvas.create_oval(
                layout.x - r,
                layout.y - r,
                layout.x + r,
                layout.y + r,
                fill=color,
                outline="#212529",
                width=2,
            )
            self._region_item_to_id[item] = region_id

            mean_activity = 0.0
            if region.neurons:
                mean_activity = sum(n.activity for n in region.neurons.values()) / len(region.neurons)
            label = f"{region_id}\n{layout.kind}\nmu={mean_activity:.3f}"
            self.region_canvas.create_text(layout.x, layout.y, text=label, font=("Arial", 9))

    def _on_region_click(self, event: tk.Event) -> None:
        clicked = self.region_canvas.find_closest(event.x, event.y)
        if not clicked:
            return
        item_id = clicked[0]
        if item_id not in self._region_item_to_id:
            return
        region_id = self._region_item_to_id[item_id]
        self.selected_region_id = region_id
        self.status_var.set(f"Selected region: {region_id}")
        self._draw_region_detail(region_id)

    def _draw_region_detail(self, region_id: str) -> None:
        self.detail_canvas.delete("all")
        region = self.network.regions[region_id]

        w = max(self.detail_canvas.winfo_width(), 740)
        h = max(self.detail_canvas.winfo_height(), 800)

        self.detail_canvas.create_text(
            16,
            14,
            anchor="nw",
            text=f"Region: {region_id}",
            font=("Arial", 14, "bold"),
            fill="#212529",
        )

        feed_in = sorted(region.feed_in_ids)
        feed_out = sorted(region.feed_out_ids)
        middle = sorted(set(region.neurons.keys()) - set(feed_in) - set(feed_out))

        def place_column(ids: List[str], x: float) -> Dict[str, Tuple[float, float]]:
            pos: Dict[str, Tuple[float, float]] = {}
            if not ids:
                return pos
            gap = (h - 80) / (len(ids) + 1)
            for i, nid in enumerate(ids, start=1):
                pos[nid] = (x, 50 + i * gap)
            return pos

        positions: Dict[str, Tuple[float, float]] = {}
        positions.update(place_column(feed_in, w * 0.18))
        positions.update(place_column(middle, w * 0.50))
        positions.update(place_column(feed_out, w * 0.82))

        # Draw internal edges first.
        for src_id, src_neuron in region.neurons.items():
            if src_id not in positions:
                continue
            x1, y1 = positions[src_id]
            for dst_id, weight in src_neuron.terminal_weights.items():
                if dst_id not in region.neurons or dst_id not in positions:
                    continue
                x2, y2 = positions[dst_id]
                self.detail_canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    arrow=tk.LAST,
                    fill="#6c757d",
                    width=1.5,
                )
                mx = (x1 + x2) / 2
                my = (y1 + y2) / 2
                self.detail_canvas.create_text(
                    mx,
                    my - 8,
                    text=f"{weight:.2f}",
                    font=("Arial", 8),
                    fill="#495057",
                )

        # Draw neuron nodes + labels.
        node_radius = 18
        for neuron_id, (x, y) in positions.items():
            neuron = region.neurons[neuron_id]
            ntype = type(neuron).__name__
            color = NEURON_COLORS.get(ntype, NEURON_COLORS["BaseNeuron"])
            self.detail_canvas.create_oval(
                x - node_radius,
                y - node_radius,
                x + node_radius,
                y + node_radius,
                fill=color,
                outline="#212529",
                width=1.5,
            )
            short_id = neuron_id.split(":")[-1]
            self.detail_canvas.create_text(
                x,
                y,
                text=f"{short_id}\n{_safe_activity(neuron.activity)}",
                font=("Arial", 8),
            )

        # Column headers and legend.
        self.detail_canvas.create_text(w * 0.18, 34, text="feed_in", font=("Arial", 10, "bold"))
        self.detail_canvas.create_text(w * 0.50, 34, text="internal", font=("Arial", 10, "bold"))
        self.detail_canvas.create_text(w * 0.82, 34, text="feed_out", font=("Arial", 10, "bold"))

        self.detail_canvas.create_text(
            16,
            h - 70,
            anchor="nw",
            text="Neuron colors: sensory / standard / effector",
            font=("Arial", 9),
            fill="#343a40",
        )

    def _advance(self, ticks: int) -> None:
        self.step_fn(ticks)
        self.time_step += ticks
        self.header_var.set(self._header_text())
        self._draw_region_graph()
        if self.selected_region_id is not None and self.selected_region_id in self.network.regions:
            self._draw_region_detail(self.selected_region_id)
        self.status_var.set(f"Advanced by {ticks} step(s).")

    def run(self) -> None:
        self.root.mainloop()


def create_visualizer_for_runtime(runtime: Any) -> NetworkVisualizer:
    """
    Build a visualizer from a runtime object that exposes:
    - runtime.network
    - runtime.step(ticks: int)
    """
    if not hasattr(runtime, "network"):
        raise ValueError("Runtime must expose a 'network' attribute")
    if not hasattr(runtime, "step"):
        raise ValueError("Runtime must expose a step(ticks) method")
    return NetworkVisualizer(network=runtime.network, step_fn=runtime.step)


def maybe_run_visualizer(enabled: bool, runtime: Any) -> Optional[NetworkVisualizer]:
    """
    Optional runtime hook to launch visualizer in train/experiment sessions.
    """
    if not enabled:
        return None
    vis = create_visualizer_for_runtime(runtime)
    vis.run()
    return vis
