<script lang="ts">
  import { onMount } from "svelte";
  import { marked } from "marked";
  import markedKatex from "marked-katex-extension";
  import "katex/dist/katex.min.css";
  import { listDocs, readDoc, saveDoc, simConstants, type DocEntry } from "$lib/api";

  // Tokenize $…$ / $$…$$ as math BEFORE markdown parsing, so LaTeX (underscores, braces) survives
  // intact instead of being mangled by the markdown lexer. `nonStandard` allows `$x$` without the
  // usual surrounding whitespace, which the LaTeX-heavy chapters rely on.
  marked.use(markedKatex({ throwOnError: false, nonStandard: true }));

  let docs = $state<DocEntry[]>([]);
  let selected = $state<string | null>(null);
  let content = $state("");
  let savedContent = $state("");
  let status = $state("");
  let error = $state("");
  let constants = $state<Record<string, number>>({});

  const dirty = $derived(selected !== null && content !== savedContent);
  const previewHtml = $derived.by(() => {
    try {
      return marked.parse(content) as string;
    } catch (e) {
      return `<pre class="err">preview error: ${String(e)}</pre>`;
    }
  });

  const docsIn = (dir: string) => docs.filter((d) => d.dir === dir);

  async function refresh() {
    try {
      docs = await listDocs();
    } catch (e) {
      error = `could not list docs: ${e}`;
    }
  }

  async function open(path: string) {
    if (dirty && !confirm("Discard unsaved changes?")) return;
    try {
      error = "";
      const text = await readDoc(path);
      content = text;
      savedContent = text;
      selected = path;
      status = "";
    } catch (e) {
      error = `could not open ${path}: ${e}`;
    }
  }

  async function save() {
    if (selected === null || !dirty) return;
    try {
      await saveDoc(selected, content);
      savedContent = content;
      status = `saved ${selected}`;
      error = "";
    } catch (e) {
      error = `could not save: ${e}`;
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      save();
    }
  }

  onMount(async () => {
    await refresh();
    try {
      constants = await simConstants();
    } catch {
      /* sim link is optional for the docs pillar */
    }
  });
</script>

<svelte:window onkeydown={onKeydown} />

<div class="app">
  <aside class="sidebar">
    <div class="brand">neural · dashboard</div>
    {#each ["docs", "notes"] as dir}
      <div class="group">
        <div class="group-label">{dir}/</div>
        {#each docsIn(dir) as d (d.path)}
          <button
            class="doc"
            class:active={selected === d.path}
            onclick={() => open(d.path)}
            title={d.path}
          >
            {d.name}
          </button>
        {/each}
        {#if docsIn(dir).length === 0}
          <div class="empty">— none —</div>
        {/if}
      </div>
    {/each}
    <div class="sidebar-foot">
      {#if Object.keys(constants).length}
        sim linked · MSLR={constants.MSLR} · α-boost={constants.ALPHA_BOOST}
      {:else}
        sim not linked
      {/if}
    </div>
  </aside>

  <main class="main">
    <header class="toolbar">
      <span class="path">{selected ?? "no document"}{dirty ? " ●" : ""}</span>
      <span class="spacer"></span>
      {#if error}
        <span class="error">{error}</span>
      {:else if status}
        <span class="status">{status}</span>
      {/if}
      <button class="save" disabled={!dirty} onclick={save}>Save ⌘S</button>
    </header>

    {#if selected === null}
      <div class="placeholder">Select a document from the sidebar to edit.</div>
    {:else}
      <div class="panes">
        <textarea
          class="editor"
          bind:value={content}
          spellcheck="false"
          placeholder="# Markdown + $\LaTeX$ …"
        ></textarea>
        <article class="preview">{@html previewHtml}</article>
      </div>
    {/if}
  </main>
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
    display: grid;
    grid-template-columns: 240px 1fr;
    height: 100vh;
  }

  .sidebar {
    display: flex;
    flex-direction: column;
    background: #15171c;
    border-right: 1px solid #2a2d35;
    overflow-y: auto;
  }
  .brand {
    padding: 14px 16px;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #8ab4f8;
    border-bottom: 1px solid #2a2d35;
  }
  .group {
    padding: 8px 0;
  }
  .group-label {
    padding: 6px 16px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #6b7080;
  }
  .doc {
    display: block;
    width: 100%;
    text-align: left;
    padding: 6px 16px 6px 22px;
    background: none;
    border: none;
    color: #c2c6cf;
    font: inherit;
    font-size: 13px;
    cursor: pointer;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .doc:hover {
    background: #21242c;
  }
  .doc.active {
    background: #243044;
    color: #cfe0ff;
  }
  .empty {
    padding: 4px 22px;
    font-size: 12px;
    color: #555a66;
  }
  .sidebar-foot {
    margin-top: auto;
    padding: 10px 16px;
    font-size: 11px;
    color: #6b7080;
    border-top: 1px solid #2a2d35;
  }

  .main {
    display: flex;
    flex-direction: column;
    min-width: 0;
  }
  .toolbar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 14px;
    border-bottom: 1px solid #2a2d35;
    background: #1e2128;
  }
  .path {
    font-size: 13px;
    color: #9aa0ad;
  }
  .spacer {
    flex: 1;
  }
  .status {
    font-size: 12px;
    color: #7fd18a;
  }
  .error {
    font-size: 12px;
    color: #f08a8a;
  }
  .save {
    font: inherit;
    font-size: 12px;
    padding: 5px 12px;
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

  .placeholder {
    margin: auto;
    color: #6b7080;
  }

  .panes {
    display: grid;
    grid-template-columns: 1fr 1fr;
    flex: 1;
    min-height: 0;
  }
  .editor {
    border: none;
    outline: none;
    resize: none;
    padding: 16px 18px;
    background: #16181d;
    color: #d7dae0;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 13px;
    line-height: 1.6;
    border-right: 1px solid #2a2d35;
  }
  .preview {
    padding: 8px 24px 40px;
    overflow-y: auto;
    line-height: 1.6;
  }
  .preview :global(h1),
  .preview :global(h2),
  .preview :global(h3) {
    color: #eaecef;
    border-bottom: 1px solid #2a2d35;
    padding-bottom: 0.2em;
  }
  .preview :global(a) {
    color: #8ab4f8;
  }
  .preview :global(code) {
    background: #23262e;
    padding: 0.1em 0.35em;
    border-radius: 4px;
    font-size: 0.9em;
  }
  .preview :global(pre) {
    background: #16181d;
    border: 1px solid #2a2d35;
    border-radius: 6px;
    padding: 12px;
    overflow-x: auto;
  }
  .preview :global(pre code) {
    background: none;
    padding: 0;
  }
  .preview :global(table) {
    border-collapse: collapse;
  }
  .preview :global(th),
  .preview :global(td) {
    border: 1px solid #2a2d35;
    padding: 4px 10px;
  }
  .preview :global(blockquote) {
    border-left: 3px solid #3a4a66;
    margin-left: 0;
    padding-left: 14px;
    color: #9aa0ad;
  }
</style>
