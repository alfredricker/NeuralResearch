<script lang="ts">
  import type { NeuronAnatomy, NeuronFrame } from "$lib/api";

  // The L2 view: draw ONE neuron *as* a neuron — soma at center, basal dendrites fanning down,
  // apical dendrites fanning up, synapses as beads placed along each branch by their `x` (0..255).
  // Anatomy is the fixed skeleton; an optional frame paints the live channels over it (branch V_B,
  // soma potential/beta, synapse weight/alpha, and the per-step firing flashes). Clicking a synapse
  // bead injects an AP onto that slot via `onStimulate` — the primitive that drives an isolated
  // neuron with no upstream population wired in.
  interface Props {
    anatomy: NeuronAnatomy;
    frame?: NeuronFrame | null;
    onStimulate: (synapse: number) => void;
  }
  let { anatomy, frame = null, onStimulate }: Props = $props();

  // --- canvas geometry ---
  const W = 620;
  const H = 620;
  const CX = W / 2;
  const CY = H / 2;
  const SOMA_R = 30;
  const BRANCH_LEN = 250; // soma edge → branch tip
  const FAN = 130; // total angular spread of a hemisphere's dendrites, degrees

  const clamp01 = (t: number) => Math.max(0, Math.min(1, t));
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

  // Cool→warm ramp (blue → orange) for any normalized activation channel.
  function warmCool(t: number): string {
    t = clamp01(t);
    return `hsl(${lerp(212, 20, t).toFixed(0)} ${lerp(50, 88, t).toFixed(0)}% ${lerp(58, 56, t).toFixed(0)}%)`;
  }

  const deg = (d: number) => (d * Math.PI) / 180;

  // Split dendrites into the two hemispheres. Apical fan up (screen −y), basal fan down (+y);
  // each is spread evenly across `FAN` degrees centered on its pole.
  type Branch = {
    dendrite: number;
    is_apical: boolean;
    angle: number; // screen radians: 0 = right, +y = down
    x1: number; y1: number; x2: number; y2: number;
    vNorm: number; // V_B / threshold, clamped
    fired: boolean;
    beads: Array<{
      synapse: number;
      cx: number; cy: number;
      bound: boolean;
      wNorm: number; // weight magnitude, normalized
      haloR: number; // scaled by alpha
      signaled: boolean;
    }>;
  };

  const layout = $derived.by<Branch[]>(() => {
    const basal = anatomy.dendrites.filter((d) => !d.is_apical);
    const apical = anatomy.dendrites.filter((d) => d.is_apical);

    const place = (
      list: typeof anatomy.dendrites,
      poleDeg: number,
    ): Branch[] =>
      list.map((d, i) => {
        const spread = list.length > 1 ? FAN / (list.length - 1) : 0;
        const angle = deg(poleDeg - FAN / 2 + spread * i);
        const ux = Math.cos(angle);
        const uy = Math.sin(angle);
        const x1 = CX + ux * SOMA_R;
        const y1 = CY + uy * SOMA_R;
        const x2 = CX + ux * (SOMA_R + BRANCH_LEN);
        const y2 = CY + uy * (SOMA_R + BRANCH_LEN);

        // Find this dendrite's live state in the parallel frame array (same order as anatomy).
        const fd = frame?.dendrites[anatomy.dendrites.indexOf(d)] ?? null;
        const vNorm = clamp01((fd?.v_b ?? 0) / Math.max(1, d.threshold));

        const beads = d.synapses.map((s, j) => {
          const t = s.x / 255;
          const r = SOMA_R + BRANCH_LEN * t;
          const fs = fd?.synapses[j] ?? null;
          const wNorm = clamp01(Math.abs(fs?.weight ?? 0) / 12);
          const haloR = 7 + ((fs?.alpha ?? 0) / 255) * 16;
          return {
            synapse: s.synapse,
            cx: CX + ux * r,
            cy: CY + uy * r,
            bound: s.src_neuron !== null,
            wNorm,
            haloR,
            signaled: fs?.signaled ?? false,
          };
        });

        return { dendrite: d.dendrite, is_apical: d.is_apical, angle, x1, y1, x2, y2, vNorm, fired: fd?.fired ?? false, beads };
      });

    return [...place(apical, 270), ...place(basal, 90)];
  });

  // --- soma channels ---
  const somaT = $derived(clamp01((frame?.soma_potential ?? 0) / Math.max(1, anatomy.soma_threshold)));
  const betaT = $derived(clamp01((frame?.soma_beta ?? 0) / 63)); // 6-bit burst counter
  const bursting = $derived((frame?.soma_burst ?? 0) > 0);
  // Beta gauge: a ring arc that fills clockwise from the top.
  const BETA_R = SOMA_R + 9;
  const betaCirc = 2 * Math.PI * BETA_R;
</script>

<svg viewBox="0 0 {W} {H}" class="neuron" role="img" aria-label="single neuron, layer 2 view">
  <!-- dendrite branches (drawn under the soma & beads) -->
  {#each layout as b (b.dendrite)}
    <line
      x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2}
      stroke={warmCool(b.vNorm)}
      stroke-width={b.fired ? 9 : 4}
      stroke-linecap="round"
      opacity={b.is_apical ? 0.95 : 0.85}
      class:fired={b.fired}
    />
    {#if b.fired}
      <line x1={b.x1} y1={b.y1} x2={b.x2} y2={b.y2} stroke="#fff6d6" stroke-width="2" stroke-linecap="round" class="flash-line" />
    {/if}
  {/each}

  <!-- synapse beads -->
  {#each layout as b (b.dendrite)}
    {#each b.beads as bead (bead.synapse)}
      <!-- alpha halo -->
      <circle cx={bead.cx} cy={bead.cy} r={bead.haloR} fill={warmCool(bead.wNorm)} opacity="0.18" />
      <!-- signaled pulse ring -->
      {#if bead.signaled}
        <circle cx={bead.cx} cy={bead.cy} r={bead.haloR + 3} fill="none" stroke="#fff6d6" stroke-width="2.5" class="pulse" />
      {/if}
      <!-- bead: fill = weight, dashed outline if unbound (directly stimulable) -->
      <circle
        cx={bead.cx} cy={bead.cy} r="7"
        fill={warmCool(bead.wNorm)}
        stroke={bead.bound ? "#0d0f14" : "#aeb6c6"}
        stroke-width="1.5"
        stroke-dasharray={bead.bound ? "none" : "2 2"}
        class="bead"
        role="button"
        tabindex="0"
        aria-label="stimulate synapse {bead.synapse}"
        onclick={() => onStimulate(bead.synapse)}
        onkeydown={(e) => (e.key === "Enter" || e.key === " ") && onStimulate(bead.synapse)}
      />
    {/each}
  {/each}

  <!-- soma -->
  {#if bursting}
    <circle cx={CX} cy={CY} r={SOMA_R + 16} fill="#fff6d6" opacity="0.5" class="flash-line" />
  {/if}
  <circle cx={CX} cy={CY} r={SOMA_R} fill={warmCool(somaT)} stroke="#0d0f14" stroke-width="2" />
  <!-- beta gauge ring -->
  <circle
    cx={CX} cy={CY} r={BETA_R}
    fill="none" stroke="#2a2d35" stroke-width="4"
  />
  <circle
    cx={CX} cy={CY} r={BETA_R}
    fill="none" stroke="#8ab4f8" stroke-width="4" stroke-linecap="round"
    stroke-dasharray="{(betaT * betaCirc).toFixed(1)} {betaCirc.toFixed(1)}"
    transform="rotate(-90 {CX} {CY})"
  />
</svg>

<style>
  .neuron {
    width: 100%;
    height: 100%;
    display: block;
  }
  .bead {
    cursor: pointer;
    transition: r 0.1s ease;
  }
  .bead:hover {
    r: 9;
  }
  .bead:focus {
    outline: none;
  }
  .fired {
    transition: stroke-width 0.08s ease;
  }
  .flash-line {
    animation: fade 0.45s ease-out forwards;
  }
  .pulse {
    animation: pulse 0.5s ease-out forwards;
  }
  @keyframes fade {
    from { opacity: 0.9; }
    to { opacity: 0; }
  }
  @keyframes pulse {
    from { opacity: 0.9; stroke-width: 3; }
    to { opacity: 0; stroke-width: 8; }
  }
</style>
