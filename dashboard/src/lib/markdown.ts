// Shared markdown + KaTeX rendering, configured once for the whole app.
//
// We tokenize $…$ / $$…$$ as math BEFORE the markdown lexer runs, so LaTeX (underscores, braces)
// survives intact instead of being mangled. `nonStandard` allows `$x$` without the usual surrounding
// whitespace, which the LaTeX-heavy chapters rely on.
import { marked } from "marked";
import markedKatex from "marked-katex-extension";
import "katex/dist/katex.min.css";

marked.use(markedKatex({ throwOnError: false, nonStandard: true }));

/** Render markdown (with inline/block LaTeX) to HTML, never throwing — errors render as a notice. */
export function renderMarkdown(src: string): string {
  try {
    return marked.parse(src) as string;
  } catch (e) {
    return `<pre class="md-err">preview error: ${String(e)}</pre>`;
  }
}
