<script lang="ts">
  import { onMount } from "svelte";
  import {
    listRecordings,
    loadRecording,
    readRunNote,
    saveRunNote,
    runNotePath,
    type RecordingSummary,
    type RecordingDetail,
  } from "$lib/api";
  import { renderMarkdown } from "$lib/markdown";
  import BarChart from "./BarChart.svelte";
  import LineChart from "./LineChart.svelte";
  import NetworkView from "./NetworkView.svelte";

  type VizTab = "classification" | "network";
  let vizTab = $state<VizTab>("classification");

  let recordings = $state<RecordingSummary[]>([]);
  let selectedStem = $state<string | null>(null);
  let detail = $state<RecordingDetail | null>(null);
  let loading = $state(false);
  let error = $state("");

  // right-side run-notes panel
  let notesOpen = $state(true);
  let notePreview = $state(false);
  let note = $state("");
  let savedNote = $state("");
  let noteStatus = $state("");

  const noteDirty = $derived(detail !== null && note !== savedNote);
  const digitLabels = $derived(detail ? detail.output_spikes.map((_, i) => String(i)) : []);
  const spikePoints = $derived(detail ? detail.spikes_over_time.map((t) => ({ x: t.tick, y: t.spikes })) : []);
  const hiddenPct = $derived(
    detail && detail.hidden_count > 0 ? (100 * detail.hidden_active) / detail.hidden_count : 0,
  );

  async function refresh() {
    try {
      recordings = await listRecordings();
    } catch (e) {
      error = `could not list recordings: ${e}`;
    }
  }

  async function select(stem: string) {
    if (noteDirty && !confirm("Discard unsaved run notes?")) return;
    selectedStem = stem;
    loading = true;
    error = "";
    noteStatus = "";
    try {
      detail = await loadRecording(stem);
      const text = await readRunNote(detail.label);
      note = text;
      savedNote = text;
    } catch (e) {
      error = `could not load ${stem}: ${e}`;
      detail = null;
    } finally {
      loading = false;
    }
  }

  async function saveNote() {
    if (detail === null || !noteDirty) return;
    try {
      await saveRunNote(detail.label, note);
      savedNote = note;
      noteStatus = `saved ${runNotePath(detail.label)}`;
    } catch (e) {
      noteStatus = `save failed: ${e}`;
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      saveNote();
    }
  }

  const verdict = (r: { correct: boolean | null }) =>
    r.correct === null ? "" : r.correct ? "ok" : "miss";

  onMount(refresh);
</script>

<svelte:window onkeydown={onKeydown} />

<div class="sim">
  <!-- recording list -->
  <aside class="reclist">
    <header class="phead">
      <span class="ptitle">recordings</span>
      <button class="refresh" onclick={refresh} title="Rescan recordings/">⟳</button>
    </header>
    <div class="rows">
      {#each recordings as r (r.stem)}
        <button class="rec" class:active={selectedStem === r.stem} onclick={() => select(r.stem)}>
          <span class="rec-label">{r.label}</span>
          <span class="rec-pred">
            {#if r.true_label !== null}<span class="truth">t{r.true_label}</span>{/if}
            {#if r.prediction !== null}<span class="pred">→ p{r.prediction}</span>{/if}
          </span>
          {#if verdict(r)}<span class="chip {verdict(r)}">{verdict(r)}</span>{/if}
        </button>
      {/each}
      {#if recordings.length === 0}
        <div class="empty">no recordings — run neural-cli to generate some</div>
      {/if}
    </div>
  </aside>

  <!-- viz dashboard -->
  <main class="viz">
    {#if error}
      <div class="banner err">{error}</div>
    {/if}
    {#if loading}
      <div class="placeholder">loading…</div>
    {:else if detail === null}
      <div class="placeholder">Select a recording to inspect its trial.</div>
    {:else}
      <div class="viztabs">
        <button class:active={vizTab === "classification"} onclick={() => (vizTab = "classification")}>Classification</button>
        <button class:active={vizTab === "network"} onclick={() => (vizTab = "network")}>Network</button>
      </div>
      {#if vizTab === "network"}
        <NetworkView stem={selectedStem} />
      {:else}
      <div class="scroll">
        <!-- summary + prediction -->
        <section class="card">
          <div class="card-head">
            <h2>{detail.label}</h2>
            {#if detail.correct !== null}
              <span class="chip {detail.correct ? 'ok' : 'miss'}">
                {detail.correct ? "correct" : "wrong"}
              </span>
            {/if}
          </div>
          <div class="verdict-row">
            <div class="stat"><span class="k">true label</span><span class="v">{detail.true_label ?? "—"}</span></div>
            <div class="stat"><span class="k">prediction</span><span class="v">{detail.prediction ?? "silent"}</span></div>
            <div class="stat"><span class="k">events</span><span class="v">{detail.event_total.toLocaleString()}</span></div>
            <div class="stat"><span class="k">somatic spikes</span><span class="v">{detail.somatic_total.toLocaleString()}</span></div>
          </div>
        </section>

        <!-- output read-out -->
        <section class="card">
          <h3>output read-out · spikes per digit</h3>
          <BarChart values={detail.output_spikes} labels={digitLabels} highlight={detail.prediction} truth={detail.true_label} />
          <p class="legend">
            <span class="sw pred"></span> prediction (argmax)
            <span class="sw truth"></span> true label
            <span class="sw ok"></span> both (correct)
          </p>
        </section>

        <!-- spikes over time -->
        <section class="card">
          <h3>somatic spikes over time</h3>
          <LineChart points={spikePoints} />
        </section>

        <!-- layer activity + topology -->
        <section class="card grid2">
          <div>
            <h3>layer activity (post-trial)</h3>
            <table class="kv">
              <tbody>
                <tr><td>input spikes</td><td>{detail.input_total.toLocaleString()}</td></tr>
                <tr><td>hidden spikes</td><td>{detail.hidden_total.toLocaleString()}</td></tr>
                <tr><td>output spikes</td><td>{detail.output_total.toLocaleString()}</td></tr>
                <tr>
                  <td>hidden active</td>
                  <td>{detail.hidden_active} / {detail.hidden_count} ({hiddenPct.toFixed(0)}%)</td>
                </tr>
              </tbody>
            </table>
          </div>
          <div>
            <h3>topology</h3>
            <table class="kv">
              <tbody>
                {#each Object.entries(detail.dims) as [k, v]}
                  <tr><td>{k}</td><td>{v.toLocaleString()}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
        </section>

        <!-- constants -->
        <section class="card">
          <h3>constants</h3>
          <div class="consts">
            {#each Object.entries(detail.constants) as [k, v]}
              <span class="const"><span class="ck">{k}</span><span class="cv">{v}</span></span>
            {/each}
          </div>
        </section>
      </div>
      {/if}
    {/if}
  </main>

  <!-- run-notes panel -->
  {#if notesOpen}
    <aside class="notes">
      <header class="phead">
        <span class="ptitle">run notes</span>
        <span class="spacer"></span>
        <button class="ghost" class:on={notePreview} onclick={() => (notePreview = !notePreview)} disabled={detail === null}>preview</button>
        <button class="save" disabled={!noteDirty} onclick={saveNote}>Save ⌘S</button>
        <button class="collapse" onclick={() => (notesOpen = false)} title="Collapse">›</button>
      </header>
      {#if detail === null}
        <div class="placeholder small">Select a recording to take notes on it.</div>
      {:else}
        <div class="note-path">{runNotePath(detail.label)}{noteDirty ? " ●" : ""}</div>
        {#if notePreview}
          <article class="preview">{@html renderMarkdown(note)}</article>
        {:else}
          <textarea class="note-edit" bind:value={note} spellcheck="false" placeholder={`# ${detail.label}\n\nObservations, hypotheses, next steps…  (Markdown + $\\LaTeX$)`}></textarea>
        {/if}
        {#if noteStatus}<div class="note-status">{noteStatus}</div>{/if}
      {/if}
    </aside>
  {:else}
    <button class="strip" onclick={() => (notesOpen = true)} title="Expand run notes">
      <span class="chev">‹</span><span class="vlabel">run notes</span>
    </button>
  {/if}
</div>

<style>
  .sim {
    display: flex;
    height: 100%;
    min-height: 0;
  }
  .phead {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px 7px 14px;
    border-bottom: 1px solid #2a2d35;
    background: #1e2128;
  }
  .ptitle {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 10px;
    color: #6b7080;
  }
  .spacer {
    flex: 1;
  }

  /* recording list */
  .reclist {
    width: 230px;
    flex: none;
    display: flex;
    flex-direction: column;
    background: #15171c;
    border-right: 1px solid #2a2d35;
    min-height: 0;
  }
  .refresh {
    margin-left: auto;
    font: inherit;
    background: none;
    border: none;
    color: #8a909d;
    cursor: pointer;
    font-size: 14px;
  }
  .refresh:hover {
    color: #cfd6e4;
  }
  .rows {
    overflow-y: auto;
  }
  .rec {
    display: flex;
    align-items: center;
    gap: 6px;
    width: 100%;
    text-align: left;
    padding: 8px 12px;
    background: none;
    border: none;
    border-bottom: 1px solid #1d2027;
    color: #c2c6cf;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
  }
  .rec:hover {
    background: #21242c;
  }
  .rec.active {
    background: #243044;
  }
  .rec-label {
    flex: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .rec-pred {
    font-size: 11px;
    color: #7a808d;
  }
  .truth {
    color: #6f8fd6;
  }
  .pred {
    color: #cf9a5a;
  }

  .chip {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .chip.ok {
    background: #1f3a26;
    color: #7fd18a;
  }
  .chip.miss {
    background: #3a2424;
    color: #e89a9a;
  }

  /* viz */
  .viz {
    flex: 1;
    min-width: 0;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .viztabs {
    display: flex;
    gap: 2px;
    padding: 8px 14px 0;
    border-bottom: 1px solid #2a2d35;
    background: #1a1d23;
  }
  .viztabs button {
    font: inherit;
    font-size: 13px;
    padding: 6px 14px;
    border: none;
    border-bottom: 2px solid transparent;
    background: none;
    color: #9aa0ad;
    cursor: pointer;
  }
  .viztabs button:hover {
    color: #cfd6e4;
  }
  .viztabs button.active {
    color: #cfe0ff;
    border-bottom-color: #6f8fd6;
  }
  .scroll {
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    min-height: 0;
  }
  .card {
    background: #1a1d23;
    border: 1px solid #2a2d35;
    border-radius: 8px;
    padding: 14px 16px;
  }
  .card.grid2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .card-head {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .card h2 {
    margin: 0;
    font-size: 18px;
    color: #eaecef;
  }
  .card h3 {
    margin: 0 0 10px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8a909d;
    font-weight: 600;
  }
  .verdict-row {
    display: flex;
    gap: 28px;
    margin-top: 12px;
  }
  .stat {
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .stat .k {
    font-size: 11px;
    color: #6b7080;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .stat .v {
    font-size: 18px;
    color: #d7dae0;
  }
  .legend {
    margin: 10px 0 0;
    font-size: 11px;
    color: #7a808d;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sw {
    display: inline-block;
    width: 11px;
    height: 11px;
    border-radius: 3px;
    margin-left: 12px;
  }
  .sw.pred {
    background: #f0a35a;
  }
  .sw.truth {
    background: #5a7fd1;
  }
  .sw.ok {
    background: #7fd18a;
  }
  .kv {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
  }
  .kv td {
    padding: 4px 0;
    border-bottom: 1px solid #23262e;
    color: #c2c6cf;
  }
  .kv td:last-child {
    text-align: right;
    color: #e1e4ea;
    font-variant-numeric: tabular-nums;
  }
  .consts {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .const {
    display: inline-flex;
    gap: 6px;
    background: #16181d;
    border: 1px solid #2a2d35;
    border-radius: 5px;
    padding: 3px 8px;
    font-size: 12px;
  }
  .ck {
    color: #7a808d;
  }
  .cv {
    color: #cfd6e4;
    font-variant-numeric: tabular-nums;
  }

  .placeholder {
    margin: auto;
    color: #6b7080;
    font-size: 13px;
  }
  .placeholder.small {
    margin: 24px 14px;
    font-size: 12px;
  }
  .banner {
    padding: 8px 14px;
    font-size: 12px;
  }
  .banner.err {
    background: #3a2424;
    color: #e89a9a;
  }
  .empty {
    padding: 14px;
    font-size: 12px;
    color: #555a66;
  }

  /* run notes */
  .notes {
    width: 340px;
    flex: none;
    display: flex;
    flex-direction: column;
    background: #15171c;
    border-left: 1px solid #2a2d35;
    min-height: 0;
  }
  .note-path {
    padding: 6px 12px;
    font-size: 11px;
    color: #6b7080;
    border-bottom: 1px solid #23262e;
  }
  .note-edit {
    flex: 1;
    border: none;
    outline: none;
    resize: none;
    padding: 12px 14px;
    background: #16181d;
    color: #d7dae0;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 13px;
    line-height: 1.6;
    min-height: 0;
  }
  .preview {
    flex: 1;
    padding: 8px 16px 32px;
    overflow-y: auto;
    line-height: 1.6;
    min-height: 0;
    font-size: 14px;
  }
  .preview :global(h1),
  .preview :global(h2) {
    color: #eaecef;
    border-bottom: 1px solid #2a2d35;
    padding-bottom: 0.2em;
  }
  .preview :global(code) {
    background: #23262e;
    padding: 0.1em 0.35em;
    border-radius: 4px;
  }
  .note-status {
    padding: 6px 12px;
    font-size: 11px;
    color: #7fd18a;
    border-top: 1px solid #23262e;
  }

  .save {
    font: inherit;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #3a4a66;
    background: #243044;
    color: #cfe0ff;
    cursor: pointer;
  }
  .save:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .ghost {
    font: inherit;
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 6px;
    border: 1px solid #2a2d35;
    background: #20232b;
    color: #9aa0ad;
    cursor: pointer;
  }
  .ghost.on {
    background: #243044;
    color: #cfe0ff;
    border-color: #3a4a66;
  }
  .ghost:disabled {
    opacity: 0.4;
    cursor: default;
  }
  .collapse {
    font: inherit;
    font-size: 14px;
    line-height: 1;
    width: 22px;
    height: 22px;
    border-radius: 5px;
    border: 1px solid #2a2d35;
    background: #20232b;
    color: #9aa0ad;
    cursor: pointer;
  }
  .collapse:hover {
    background: #272b34;
  }
  .strip {
    width: 32px;
    flex: none;
    border: none;
    border-left: 1px solid #2a2d35;
    background: #15171c;
    color: #8a909d;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding-top: 10px;
  }
  .strip:hover {
    background: #1b1e25;
    color: #c2c6cf;
  }
  .chev {
    font-size: 14px;
  }
  .vlabel {
    writing-mode: vertical-rl;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 10px;
  }
</style>
