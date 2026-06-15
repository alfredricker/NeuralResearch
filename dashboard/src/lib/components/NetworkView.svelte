<script lang="ts">
  import { loadNetwork, type NetworkReplay, type EdgeMeta } from "$lib/api";

  // The selected recording stem; the component loads its own topology + per-tick timeline.
  let { stem }: { stem: string | null } = $props();

  let net = $state<NetworkReplay | null>(null);
  let loading = $state(false);
  let error = $state("");

  let tickIdx = $state(0);
  let playing = $state(false);
  let selected = $state<number | null>(null);
  let colorMode = $state<"weight" | "delta">("weight");

  const W = 880;
  const H = 560;
  const LAYER_BASE = ["#3f6fa8", "#6f63c4", "#3aa88a"]; // input / hidden / output
  const POT_SCALE = 20; // ≈ soma threshold; potentials above this read as "hot"

  // (re)load whenever the selected recording changes.
  $effect(() => {
    const s = stem;
    if (!s) {
      net = null;
      return;
    }
    loading = true;
    error = "";
    selected = null;
    loadNetwork(s)
      .then((n) => {
        net = n;
        tickIdx = 0;
      })
      .catch((e) => {
        error = `${e}`;
        net = null;
      })
      .finally(() => (loading = false));
  });

  // playback: advance one tick per interval while playing, looping at the end.
  $effect(() => {
    if (!playing || !net || net.ticks.length === 0) return;
    const id = setInterval(() => {
      tickIdx = (tickIdx + 1) % net!.ticks.length;
    }, 140);
    return () => clearInterval(id);
  });

  const frame = $derived(
    net && net.has_per_tick && net.ticks.length ? net.ticks[Math.min(tickIdx, net.ticks.length - 1)] : null,
  );

  // node positions, indexed by neuron index. Input lays out as a square grid (the place-cell sheet)
  // when it's a perfect square, else a column; hidden/output are columns.
  const pos = $derived.by(() => {
    const p: { x: number; y: number; r: number }[] = [];
    if (!net) return p;
    const input: number[] = [];
    const hidden: number[] = [];
    const output: number[] = [];
    for (const nm of net.neurons) {
      if (nm.layer === 0) input.push(nm.index);
      else if (nm.layer === 2) output.push(nm.index);
      else hidden.push(nm.index);
    }
    const column = (idxs: number[], x: number, top: number, bottom: number) => {
      const k = idxs.length;
      const span = bottom - top;
      const r = Math.max(5, Math.min(13, span / (k * 2.2)));
      idxs.forEach((idx, j) => {
        const y = k === 1 ? (top + bottom) / 2 : top + ((j + 0.5) / k) * span;
        p[idx] = { x, y, r };
      });
    };
    const side = Math.round(Math.sqrt(input.length));
    if (input.length > 1 && side * side === input.length) {
      const top = 70;
      const region = Math.min(200, H - 140);
      const cell = region / side;
      const r = cell * 0.34;
      const gx = 60;
      const gy = top + (H - 140 - cell * side) / 2;
      input.forEach((idx, i) => {
        const row = Math.floor(i / side);
        const col = i % side;
        p[idx] = { x: gx + (col + 0.5) * cell, y: gy + (row + 0.5) * cell, r };
      });
    } else {
      column(input, 100, 70, H - 70);
    }
    column(hidden, W / 2, 50, H - 50);
    column(output, W - 110, 130, H - 130);
    return p;
  });

  const wmax = $derived.by(() => {
    if (!net) return 1;
    let m = 1;
    for (const e of net.edges) {
      const v = colorMode === "delta" ? e.w_post - e.w_pre : e.w_post;
      m = Math.max(m, Math.abs(v));
    }
    return m;
  });

  function edgeValue(e: EdgeMeta): number {
    return colorMode === "delta" ? e.w_post - e.w_pre : e.w_post;
  }
  function edgeStroke(e: EdgeMeta): string {
    const v = edgeValue(e);
    const dim = selected !== null && e.src !== selected && e.dst !== selected;
    const a = Math.min(0.9, 0.1 + (Math.abs(v) / wmax) * 0.85) * (dim ? 0.12 : 1);
    return v >= 0 ? `rgba(232,150,74,${a})` : `rgba(90,140,220,${a})`;
  }
  function edgeWidth(e: EdgeMeta): number {
    return selected !== null && (e.src === selected || e.dst === selected) ? 2.2 : 1;
  }

  function lerp(a: string, b: string, t: number): string {
    const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)];
    const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)];
    const c = pa.map((x, i) => Math.round(x + (pb[i] - x) * t));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }
  function nodeFill(i: number): string {
    const base = LAYER_BASE[net!.neurons[i].layer];
    if (frame) {
      if (frame.spikes[i] > 0) return "#ffd24a"; // fired this tick
      const t = Math.max(0, Math.min(1, frame.potentials[i] / POT_SCALE));
      return lerp(base, "#f0d68a", t * 0.75);
    }
    return base;
  }
  function nodeStroke(i: number): string {
    if (selected === i) return "#cfe0ff";
    if (frame && frame.spikes[i] > 0) return "#fff3c0";
    return "#11131a";
  }

  // inspector: a clicked neuron's afferents (incoming) and efferents (outgoing).
  const afferents = $derived(net && selected !== null ? net.edges.filter((e) => e.dst === selected) : []);
  const efferents = $derived(net && selected !== null ? net.edges.filter((e) => e.src === selected) : []);
  const layerName = (l: number) => ["input", "hidden", "output"][l] ?? "?";

  function step(d: number) {
    if (!net || net.ticks.length === 0) return;
    playing = false;
    tickIdx = Math.max(0, Math.min(net.ticks.length - 1, tickIdx + d));
  }
</script>

<div class="net">
  {#if loading}
    <div class="ph">loading network…</div>
  {:else if error}
    <div class="ph err">{error}</div>
  {:else if !net}
    <div class="ph">Select a recording to view its network.</div>
  {:else}
    <!-- controls -->
    <div class="controls">
      <button class="tbtn" onclick={() => step(-1)} disabled={!net.has_per_tick} title="Step back">◀</button>
      <button class="tbtn play" onclick={() => (playing = !playing)} disabled={!net.has_per_tick} title="Play / pause">
        {playing ? "⏸" : "▶"}
      </button>
      <button class="tbtn" onclick={() => step(1)} disabled={!net.has_per_tick} title="Step (one wavefront)">▶▌</button>
      {#if net.has_per_tick}
        <input
          class="slider"
          type="range"
          min="0"
          max={net.ticks.length - 1}
          bind:value={tickIdx}
          oninput={() => (playing = false)}
        />
        <span class="tickread">tick {frame?.tick ?? 0} · {tickIdx + 1}/{net.ticks.length}</span>
      {:else}
        <span class="tickread muted">no per-tick state (large net) — showing topology + final weights</span>
      {/if}
      <span class="spacer"></span>
      <div class="seg">
        <button class:active={colorMode === "weight"} onclick={() => (colorMode = "weight")}>weight</button>
        <button class:active={colorMode === "delta"} onclick={() => (colorMode = "delta")}>Δ weight</button>
      </div>
    </div>

    <div class="stage">
      <!-- graph -->
      <svg viewBox={`0 0 ${W} ${H}`} class="graph" role="img" aria-label="network graph">
        <!-- background click clears selection -->
        <rect x="0" y="0" width={W} height={H} fill="transparent" onclick={() => (selected = null)} />

        <!-- column captions -->
        <text class="cap" x="100" y="28" text-anchor="middle">input ({net.n_input})</text>
        <text class="cap" x={W / 2} y="28" text-anchor="middle">hidden ({net.n_hidden})</text>
        <text class="cap" x={W - 110} y="28" text-anchor="middle">output ({net.n_output})</text>

        <!-- edges -->
        <g>
          {#each net.edges as e (e.synapse)}
            {#if pos[e.src] && pos[e.dst]}
              <line
                x1={pos[e.src].x}
                y1={pos[e.src].y}
                x2={pos[e.dst].x}
                y2={pos[e.dst].y}
                stroke={edgeStroke(e)}
                stroke-width={edgeWidth(e)}
              />
            {/if}
          {/each}
        </g>

        <!-- nodes -->
        <g>
          {#each net.neurons as nm (nm.index)}
            {#if pos[nm.index]}
              <circle
                cx={pos[nm.index].x}
                cy={pos[nm.index].y}
                r={pos[nm.index].r + (selected === nm.index ? 2 : 0)}
                fill={nodeFill(nm.index)}
                stroke={nodeStroke(nm.index)}
                stroke-width={selected === nm.index || (frame && frame.spikes[nm.index] > 0) ? 2.4 : 1}
                class="node"
                onclick={() => (selected = nm.index)}
              />
            {/if}
          {/each}
        </g>
      </svg>

      <!-- inspector -->
      <aside class="inspect">
        {#if selected === null}
          <div class="hint">
            <p>Click a neuron to inspect it.</p>
            <p class="dim">Step the wavefront to watch potentials build and spikes propagate. Edge color is the synapse
              {colorMode === "delta" ? "weight change (this trial)" : "weight"} — warm positive, cool negative.</p>
            {#if net.edges_truncated}
              <p class="dim">Showing {net.edges.length.toLocaleString()} of {net.edge_total.toLocaleString()} edges (sampled).</p>
            {/if}
          </div>
        {:else}
          <div class="ihead">
            <span class="ititle">neuron {selected}</span>
            <span class="ilayer l{net.neurons[selected].layer}">{layerName(net.neurons[selected].layer)}</span>
          </div>
          {#if frame}
            <div class="irow"><span>soma potential</span><span class="v">{frame.potentials[selected]}</span></div>
            <div class="irow"><span>spikes this tick</span><span class="v">{frame.spikes[selected]}</span></div>
          {/if}
          <div class="irow"><span>afferents (in)</span><span class="v">{afferents.length}</span></div>
          <div class="irow"><span>efferents (out)</span><span class="v">{efferents.length}</span></div>

          {#if afferents.length}
            <h4>incoming synapses</h4>
            <div class="syntable">
              <div class="synhead"><span>from</span><span>w₀</span><span>w₁</span><span>Δ</span></div>
              {#each afferents.slice(0, 60) as e (e.synapse)}
                <button class="synrow" class:sel={selected === e.src} onclick={() => (selected = e.src)}>
                  <span>#{e.src}</span>
                  <span>{e.w_pre}</span>
                  <span>{e.w_post}</span>
                  <span class={e.w_post - e.w_pre > 0 ? "up" : e.w_post - e.w_pre < 0 ? "down" : ""}>
                    {e.w_post - e.w_pre > 0 ? "+" : ""}{e.w_post - e.w_pre}
                  </span>
                </button>
              {/each}
              {#if afferents.length > 60}<div class="more">+{afferents.length - 60} more…</div>{/if}
            </div>
          {/if}
        {/if}
      </aside>
    </div>
  {/if}
</div>

<style>
  .net {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }
  .ph {
    margin: auto;
    color: #6b7080;
    font-size: 13px;
  }
  .ph.err {
    color: #e89a9a;
  }

  .controls {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-bottom: 1px solid #2a2d35;
    background: #1a1d23;
  }
  .tbtn {
    font: inherit;
    font-size: 13px;
    min-width: 30px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid #2a2d35;
    background: #20232b;
    color: #cfd6e4;
    cursor: pointer;
  }
  .tbtn:hover:not(:disabled) {
    background: #272b34;
  }
  .tbtn:disabled {
    opacity: 0.35;
    cursor: default;
  }
  .tbtn.play {
    background: #243044;
    border-color: #3a4a66;
    color: #cfe0ff;
  }
  .slider {
    flex: 1;
    max-width: 320px;
    accent-color: #6f8fd6;
  }
  .tickread {
    font-size: 12px;
    color: #9aa0ad;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .tickread.muted {
    color: #6b7080;
  }
  .spacer {
    flex: 1;
  }
  .seg {
    display: flex;
    gap: 2px;
    background: #1e2128;
    border: 1px solid #2a2d35;
    border-radius: 7px;
    padding: 2px;
  }
  .seg button {
    font: inherit;
    font-size: 12px;
    padding: 4px 10px;
    border: none;
    border-radius: 5px;
    background: none;
    color: #9aa0ad;
    cursor: pointer;
  }
  .seg button.active {
    background: #243044;
    color: #cfe0ff;
  }

  .stage {
    flex: 1;
    display: flex;
    min-height: 0;
  }
  .graph {
    flex: 1;
    min-width: 0;
    height: 100%;
    background: #15171c;
  }
  .cap {
    fill: #6b7080;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .node {
    cursor: pointer;
  }
  .node:hover {
    stroke: #cfe0ff;
  }

  .inspect {
    width: 270px;
    flex: none;
    border-left: 1px solid #2a2d35;
    background: #1a1d23;
    padding: 14px;
    overflow-y: auto;
    font-size: 13px;
  }
  .hint p {
    margin: 0 0 10px;
    color: #c2c6cf;
  }
  .hint .dim {
    color: #7a808d;
    font-size: 12px;
  }
  .ihead {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }
  .ititle {
    font-size: 15px;
    color: #eaecef;
    font-weight: 600;
  }
  .ilayer {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .ilayer.l0 {
    background: #1f3147;
    color: #8fb6e8;
  }
  .ilayer.l1 {
    background: #2a2647;
    color: #b3a8ec;
  }
  .ilayer.l2 {
    background: #1d3a33;
    color: #7fd1b8;
  }
  .irow {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #23262e;
    color: #9aa0ad;
  }
  .irow .v {
    color: #e1e4ea;
    font-variant-numeric: tabular-nums;
  }
  h4 {
    margin: 14px 0 6px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7080;
  }
  .syntable {
    display: flex;
    flex-direction: column;
  }
  .synhead,
  .synrow {
    display: grid;
    grid-template-columns: 1fr 0.7fr 0.7fr 0.8fr;
    gap: 4px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .synhead {
    color: #6b7080;
    padding: 2px 4px;
  }
  .synrow {
    background: none;
    border: none;
    text-align: left;
    color: #c2c6cf;
    padding: 3px 4px;
    cursor: pointer;
    border-radius: 4px;
  }
  .synrow:hover {
    background: #21242c;
  }
  .synrow.sel {
    background: #243044;
  }
  .up {
    color: #7fd18a;
  }
  .down {
    color: #e89a9a;
  }
  .more {
    color: #6b7080;
    font-size: 11px;
    padding: 4px;
  }
</style>
