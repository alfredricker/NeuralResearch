<script lang="ts">
  import { onMount } from "svelte";
  import { simConstants } from "$lib/api";
  import ReadingMode from "$lib/components/ReadingMode.svelte";
  import Playground from "$lib/components/Playground.svelte";

  type Mode = "reading" | "simulation";
  let mode = $state<Mode>("reading");
  let constants = $state<Record<string, number>>({});

  onMount(async () => {
    try {
      constants = await simConstants();
    } catch {
      /* sim link is optional for the docs pillar */
    }
  });
</script>

<div class="app">
  <nav class="topbar">
    <span class="brand">neural · dashboard</span>
    <div class="modes">
      <button class:active={mode === "simulation"} onclick={() => (mode = "simulation")}>Simulation</button>
      <button class:active={mode === "reading"} onclick={() => (mode = "reading")}>Reading</button>
    </div>
    <span class="spacer"></span>
    <span class="simlink">
      {#if Object.keys(constants).length}
        sim linked · MSLR={constants.MSLR} · α-boost={constants.ALPHA_BOOST}
      {:else}
        sim not linked
      {/if}
    </span>
  </nav>

  <div class="body">
    {#if mode === "reading"}
      <ReadingMode />
    {:else}
      <Playground />
    {/if}
  </div>
</div>

<style>
  :global(html, body) {
    margin: 0;
    height: 100%;
  }
  :global(body) {
    font-family: Inter, system-ui, sans-serif;
    color: #d7dae0;
    background: #1b1d23;
  }

  .app {
    display: flex;
    flex-direction: column;
    height: 100vh;
  }
  .topbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 8px 16px;
    background: #15171c;
    border-bottom: 1px solid #2a2d35;
  }
  .brand {
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #8ab4f8;
  }
  .modes {
    display: flex;
    gap: 4px;
    background: #1e2128;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 3px;
  }
  .modes button {
    font: inherit;
    font-size: 13px;
    padding: 5px 14px;
    border: none;
    border-radius: 6px;
    background: none;
    color: #9aa0ad;
    cursor: pointer;
  }
  .modes button:hover {
    color: #cfd6e4;
  }
  .modes button.active {
    background: #243044;
    color: #cfe0ff;
  }
  .spacer {
    flex: 1;
  }
  .simlink {
    font-size: 11px;
    color: #6b7080;
  }

  .body {
    flex: 1;
    min-height: 0;
  }
</style>
