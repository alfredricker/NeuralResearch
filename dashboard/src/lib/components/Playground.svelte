<script lang="ts">
  import { onMount, onDestroy } from "svelte";
  import {
    pgBuild, pgStimulate, pgStep, pgState, pgReset,
    type NetworkSpec, type NeuronAnatomy, type NetworkFrame, type NeuronFrame,
  } from "$lib/api";
  import NeuronView from "./NeuronView.svelte";

  // A single custom neuron with both basal and apical dendrites, no upstream population — driven
  // entirely by clicking its synapses. Thresholds are tuned low enough that a few stimulations walk
  // a branch over its firing threshold, so the cascade is visible by hand. Reproducible via `seed`.
  const SINGLE_NEURON: NetworkSpec = {
    seed: 7,
    neuron_types: {
      L23: {
        n_basal_dendrites: 4,
        n_apical_dendrites: 1,
        synapse_x_sampler: { mean: 128, std: 55 },
        dendrites_per_branch: { mean: 1, std: 0 },
        synapses_per_dendrite: { mean: 5, std: 1 },
        soma_threshold: 8,
        basal_dendrite_threshold: 120,
        basal_dendrite_constant: { mean: 35, std: 15 },
        apical_dendrite_threshold: 80,
        apical_dendrite_constant: { mean: 20, std: 10 },
        learning_rate: 120,
      },
    },
    populations: [{ neuron_type: { Custom: "L23" }, size: 1, label: "neuron" }],
    connections: [],
  };

  let anatomy = $state<NeuronAnatomy | null>(null);
  let frame = $state<NeuronFrame | null>(null);
  let clock = $state(0);
  let error = $state("");
  let burst = $state(6);
  let playing = $state(false);
  let speed = $state(160); // ms per wavefront
  let timer: ReturnType<typeof setInterval> | null = null;

  function applyFrame(f: NetworkFrame) {
    clock = f.clock;
    frame = f.neurons[0] ?? null;
  }

  async function build() {
    error = "";
    stop();
    try {
      const a = await pgBuild(SINGLE_NEURON);
      anatomy = a[0] ?? null;
      applyFrame(await pgState());
    } catch (e) {
      error = `build failed: ${e}`;
    }
  }

  async function step() {
    try {
      applyFrame(await pgStep());
    } catch (e) {
      error = `step failed: ${e}`;
      stop();
    }
  }

  async function reset() {
    stop();
    try {
      await pgReset();
      applyFrame(await pgState());
    } catch (e) {
      error = `reset failed: ${e}`;
    }
  }

  async function stimulate(synapse: number) {
    try {
      await pgStimulate(synapse, burst);
      // Surface the injection immediately; the next step drains it into the dendrite.
      if (!playing) await step();
    } catch (e) {
      error = `stimulate failed: ${e}`;
    }
  }

  function play() {
    if (playing) return;
    playing = true;
    timer = setInterval(step, speed);
  }
  function stop() {
    playing = false;
    if (timer) { clearInterval(timer); timer = null; }
  }
  function toggle() {
    playing ? stop() : play();
  }

  // Restart the interval when speed changes mid-play.
  $effect(() => {
    if (playing && timer) {
      clearInterval(timer);
      timer = setInterval(step, speed);
    }
  });

  onMount(build);
  onDestroy(stop);
</script>

<div class="pg">
  <header class="bar">
    <strong>Single neuron · L2</strong>
    <span class="dim">click a synapse bead to inject an AP</span>
    <span class="spacer"></span>
    <label class="ctl">burst
      <input type="range" min="1" max="16" bind:value={burst} />
      <span class="num">{burst}</span>
    </label>
    <label class="ctl">speed
      <input type="range" min="40" max="600" step="20" bind:value={speed} />
      <span class="num">{speed}ms</span>
    </label>
    <button onclick={step} disabled={!anatomy || playing}>Step</button>
    <button class:active={playing} onclick={toggle} disabled={!anatomy}>{playing ? "Pause" : "Play"}</button>
    <button onclick={reset} disabled={!anatomy}>Reset</button>
    <button onclick={build}>Rebuild</button>
  </header>

  {#if error}
    <p class="err">{error}</p>
  {/if}

  <div class="stage">
    {#if anatomy}
      <NeuronView {anatomy} {frame} onStimulate={stimulate} />
    {:else}
      <p class="dim center">building network…</p>
    {/if}
  </div>

  <footer class="bar readout">
    <span>clock <b>{clock}</b></span>
    <span>V_soma <b>{frame?.soma_potential ?? 0}</b>/{anatomy?.soma_threshold ?? "–"}</span>
    <span>β <b>{frame?.soma_beta ?? 0}</b></span>
    <span>burst <b>{frame?.soma_burst ?? 0}</b></span>
    <span class="spacer"></span>
    <span class="legend"><i class="sw cool"></i>low <i class="sw warm"></i>high · halo=α · ring=signaled · dashed bead=unbound</span>
  </footer>
</div>

<style>
  .pg {
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
  }
  .bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    background: #15171c;
    border-bottom: 1px solid #2a2d35;
    font-size: 13px;
  }
  .readout {
    border-bottom: none;
    border-top: 1px solid #2a2d35;
    color: #9aa0ad;
    font-size: 12px;
    gap: 18px;
  }
  .readout b { color: #cfe0ff; }
  .spacer { flex: 1; }
  .dim { color: #6b7080; }
  .center { margin: auto; }
  strong { color: #cfe0ff; }
  .ctl {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #9aa0ad;
  }
  .ctl input[type="range"] { width: 90px; }
  .num { color: #cfe0ff; min-width: 34px; }
  button {
    font: inherit;
    font-size: 13px;
    padding: 5px 12px;
    border: 1px solid #2a2d35;
    border-radius: 6px;
    background: #1e2128;
    color: #cfd6e4;
    cursor: pointer;
  }
  button:hover:not(:disabled) { background: #243044; }
  button.active { background: #243044; color: #cfe0ff; }
  button:disabled { opacity: 0.4; cursor: default; }
  .err {
    margin: 0;
    padding: 6px 14px;
    background: #2c1a1a;
    color: #f0a0a0;
    font-size: 12px;
  }
  .stage {
    flex: 1;
    min-height: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at center, #20242c, #16181d);
    padding: 12px;
  }
  .legend { display: flex; align-items: center; gap: 6px; }
  .sw {
    display: inline-block;
    width: 11px; height: 11px;
    border-radius: 50%;
    vertical-align: middle;
  }
  .sw.cool { background: hsl(212 50% 58%); }
  .sw.warm { background: hsl(20 88% 56%); margin-left: 6px; }
</style>
