<script lang="ts">
  import { onMount } from "svelte";
  import { listDocs, readDoc, readDocBytes, saveDoc, renderTex, docKind, type DocEntry, type DocKind } from "$lib/api";
  import { renderMarkdown, renderLatex } from "$lib/markdown";

  // Three panels — file selection, editor, reader — each independently collapsible. Collapsing the
  // reader gives the editor the full width (focused writing); collapsing the editor does the inverse
  // (focused reading); collapsing files frees space for both.
  let collapsed = $state({ files: false, editor: false, reader: false });

  let docs = $state<DocEntry[]>([]);
  let selected = $state<string | null>(null);
  let content = $state("");
  let savedContent = $state("");
  let status = $state("");
  let error = $state("");
  // Object URL backing the PDF viewer; revoked whenever we switch away from a PDF.
  let pdfUrl = $state<string | null>(null);
  // True while a .tex compile is in flight (disables the Render button).
  let rendering = $state(false);

  const kind = $derived<DocKind | null>(selected === null ? null : docKind(selected));
  const editable = $derived(kind === "md" || kind === "tex");
  const dirty = $derived(editable && selected !== null && content !== savedContent);
  const previewHtml = $derived.by(() =>
    kind === "tex" ? renderLatex(content) : renderMarkdown(content),
  );
  const roots = ["docs", "notes", "science"] as const;
  const docsIn = (dir: string) => docs.filter((d) => d.dir === dir);

  function clearPdf() {
    if (pdfUrl) URL.revokeObjectURL(pdfUrl);
    pdfUrl = null;
  }

  async function refresh() {
    try {
      docs = await listDocs();
    } catch (e) {
      error = `could not list docs: ${e}`;
    }
  }

  // Load a document's content from disk into the editor/reader. PDFs are fetched as bytes and shown
  // via an object URL (view-only); md/tex are read as editable text.
  async function load(path: string) {
    error = "";
    status = "";
    if (docKind(path) === "pdf") {
      const bytes = await readDocBytes(path);
      clearPdf();
      pdfUrl = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
      content = "";
      savedContent = "";
    } else {
      const text = await readDoc(path);
      clearPdf();
      content = text;
      savedContent = text;
    }
    selected = path;
  }

  async function open(path: string) {
    if (dirty && !confirm("Discard unsaved changes?")) return;
    try {
      await load(path);
      if (collapsed.editor && collapsed.reader) collapsed.editor = false;
    } catch (e) {
      error = `could not open ${path}: ${e}`;
    }
  }

  // Reload the current document from disk — picks up external edits (e.g. a recompiled PDF or a run
  // note the sim just wrote). Guards unsaved editor changes.
  async function reload() {
    if (selected === null) return;
    if (dirty && !confirm("Discard unsaved changes and reload from disk?")) return;
    try {
      const path = selected;
      await load(path);
      status = `reloaded ${path}`;
    } catch (e) {
      error = `could not reload: ${e}`;
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

  // Compile the current .tex to a PDF next to it, then open that PDF. Saves first so the render
  // reflects what's in the editor.
  async function render() {
    if (selected === null || kind !== "tex" || rendering) return;
    try {
      error = "";
      if (dirty) {
        await saveDoc(selected, content);
        savedContent = content;
      }
      rendering = true;
      status = "rendering…";
      const pdfPath = await renderTex(selected);
      await refresh(); // surface the freshly written .pdf in the file tree
      await load(pdfPath); // switch the viewer to the compiled PDF
      status = `rendered ${pdfPath}`;
    } catch (e) {
      error = `render failed: ${e}`;
      status = "";
    } finally {
      rendering = false;
    }
  }

  function onKeydown(e: KeyboardEvent) {
    if ((e.metaKey || e.ctrlKey) && e.key === "s") {
      e.preventDefault();
      save();
    }
  }

  onMount(refresh);
</script>

<svelte:window onkeydown={onKeydown} />

<div class="reading">
  <!-- file selection -->
  {#if collapsed.files}
    <button class="strip" onclick={() => (collapsed.files = false)} title="Expand files">
      <span class="chev">›</span><span class="vlabel">files</span>
    </button>
  {:else}
    <section class="panel files">
      <header class="phead">
        <span class="ptitle">files</span>
        <button class="collapse" onclick={() => (collapsed.files = true)} title="Collapse">‹</button>
      </header>
      <div class="filelist">
        {#each roots as dir}
          <div class="group">
            <div class="group-label">{dir}/</div>
            {#each docsIn(dir) as d (d.path)}
              <button class="doc" class:active={selected === d.path} onclick={() => open(d.path)} title={d.path}>
                {d.name}
              </button>
            {/each}
            {#if docsIn(dir).length === 0}
              <div class="empty">— none —</div>
            {/if}
          </div>
        {/each}
      </div>
    </section>
  {/if}

  <!-- editor -->
  {#if collapsed.editor}
    <button class="strip" onclick={() => (collapsed.editor = false)} title="Expand editor">
      <span class="chev">›</span><span class="vlabel">editor</span>
    </button>
  {:else}
    <section class="panel editor-panel">
      <header class="phead">
        <span class="ptitle">editor</span>
        <span class="path">{selected ?? "no document"}{dirty ? " ●" : ""}</span>
        <span class="spacer"></span>
        {#if error}
          <span class="error">{error}</span>
        {:else if status}
          <span class="status">{status}</span>
        {/if}
        {#if kind === "tex"}
          <button class="save" disabled={rendering} onclick={render} title="Compile to PDF alongside this file">
            {rendering ? "Rendering…" : "Render PDF"}
          </button>
        {/if}
        <button class="save" disabled={!dirty} onclick={save}>Save ⌘S</button>
        <button class="collapse" onclick={() => (collapsed.editor = true)} title="Collapse">‹</button>
      </header>
      {#if selected === null}
        <div class="placeholder">Select a document to edit.</div>
      {:else if kind === "pdf"}
        <div class="placeholder">PDFs are view-only — see the reader.</div>
      {:else}
        <textarea class="editor" bind:value={content} spellcheck="false" placeholder="# Markdown + $\LaTeX$ …"></textarea>
      {/if}
    </section>
  {/if}

  <!-- reader -->
  {#if collapsed.reader}
    <button class="strip" onclick={() => (collapsed.reader = false)} title="Expand reader">
      <span class="chev">›</span><span class="vlabel">reader</span>
    </button>
  {:else}
    <section class="panel reader-panel">
      <header class="phead">
        <span class="ptitle">reader</span>
        <span class="spacer"></span>
        <button class="refresh" disabled={selected === null} onclick={reload} title="Reload from disk">⟳ Refresh</button>
        <button class="collapse" onclick={() => (collapsed.reader = true)} title="Collapse">›</button>
      </header>
      {#if selected === null}
        <div class="placeholder">Preview appears here.</div>
      {:else if kind === "pdf"}
        {#if pdfUrl}
          <iframe class="pdf" src={pdfUrl} title={selected}></iframe>
        {/if}
      {:else}
        <article class="preview">{@html previewHtml}</article>
      {/if}
    </section>
  {/if}
</div>

<style>
  .reading {
    display: flex;
    height: 100%;
    min-height: 0;
  }
  .panel {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    border-right: 1px solid #2a2d35;
  }
  .files {
    width: 240px;
    flex: none;
    background: #15171c;
  }
  .editor-panel,
  .reader-panel {
    flex: 1;
  }
  .reader-panel {
    border-right: none;
  }

  .phead {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 10px 7px 14px;
    border-bottom: 1px solid #2a2d35;
    background: #1e2128;
    font-size: 12px;
  }
  .ptitle {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 10px;
    color: #6b7080;
  }
  .path {
    color: #9aa0ad;
  }
  .spacer {
    flex: 1;
  }
  .status {
    color: #7fd18a;
  }
  .error {
    color: #f08a8a;
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
  .refresh {
    font: inherit;
    font-size: 12px;
    padding: 4px 10px;
    border-radius: 6px;
    border: 1px solid #2a2d35;
    background: #20232b;
    color: #c2c6cf;
    cursor: pointer;
  }
  .refresh:hover:not(:disabled) {
    background: #272b34;
  }
  .refresh:disabled {
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
    border-right: 1px solid #2a2d35;
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

  .filelist {
    overflow-y: auto;
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

  .placeholder {
    margin: auto;
    color: #6b7080;
    font-size: 13px;
  }
  .editor {
    flex: 1;
    border: none;
    outline: none;
    resize: none;
    padding: 16px 18px;
    background: #16181d;
    color: #d7dae0;
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 13px;
    line-height: 1.6;
    min-height: 0;
  }
  .preview {
    flex: 1;
    padding: 8px 24px 40px;
    overflow-y: auto;
    line-height: 1.6;
    min-height: 0;
  }
  .pdf {
    flex: 1;
    border: none;
    width: 100%;
    min-height: 0;
    background: #fff;
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
