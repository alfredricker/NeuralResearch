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

// Lightweight LaTeX preview. We don't run a full TeX engine — we reuse the markdown + KaTeX path by
// transforming the common document constructs (sections, math environments, \newcommand macros) into
// markdown + `$$…$$` blocks, then handing off to renderMarkdown. Anything KaTeX can't parse renders
// as an inline error (throwOnError is off), so an unsupported macro degrades gracefully.
function texToMarkdown(src: string): string {
  // Pull out the document body; if there's no \begin{document}, treat the whole thing as body.
  const docMatch = src.match(/\\begin\{document\}([\s\S]*?)\\end\{document\}/);
  const preamble = docMatch ? src.slice(0, docMatch.index) : "";
  let body = docMatch ? docMatch[1] : src;

  // Collect \newcommand / \def macros from the preamble so KaTeX can expand them. We re-emit them as
  // \newcommand prefixes inside each display block (KaTeX state doesn't persist across blocks).
  const macros: string[] = [];
  const macroRe = /\\(?:newcommand|renewcommand)\s*\{(\\[A-Za-z]+)\}(?:\[\d+\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}/g;
  for (const m of preamble.matchAll(macroRe)) macros.push(`\\newcommand{${m[1]}}{${m[2]}}`);
  const macroPrefix = macros.length ? macros.join("\n") + "\n" : "";

  // Strip line comments (a "%" not preceded by a backslash) and TeX-only commands we don't preview.
  body = body
    .replace(/(^|[^\\])%.*$/gm, "$1")
    .replace(/\\(?:maketitle|tableofcontents|noindent|centering|small|par)\b/g, "")
    .replace(/\\(?:label|ref|eqref|cite|setlength|vspace|hspace)\s*\{[^}]*\}/g, "")
    .replace(/\\(?:title|author|date)\s*\{[^}]*\}/g, "");

  // Sectioning → markdown headings.
  body = body
    .replace(/\\section\*?\s*\{([^}]*)\}/g, "\n# $1\n")
    .replace(/\\subsection\*?\s*\{([^}]*)\}/g, "\n## $1\n")
    .replace(/\\subsubsection\*?\s*\{([^}]*)\}/g, "\n### $1\n");

  // Text emphasis → markdown.
  body = body
    .replace(/\\textbf\s*\{([^}]*)\}/g, "**$1**")
    .replace(/\\(?:textit|emph)\s*\{([^}]*)\}/g, "*$1*")
    .replace(/\\texttt\s*\{([^}]*)\}/g, "`$1`");

  // Display math environments → `$$ … $$` (with macro prefix). `align`/`gather` map onto KaTeX's
  // `aligned`/`gathered`, which work inside a `$$` block.
  const display = (inner: string) => `\n\n$$\n${macroPrefix}${inner.trim()}\n$$\n\n`;
  body = body
    .replace(/\\begin\{(equation|equation\*|displaymath)\}([\s\S]*?)\\end\{\1\}/g, (_, __, inner) => display(inner))
    .replace(/\\begin\{(align|align\*|gather|gather\*)\}([\s\S]*?)\\end\{\1\}/g, (_, env, inner) => {
      const target = env.startsWith("align") ? "aligned" : "gathered";
      return display(`\\begin{${target}}\n${inner.trim()}\n\\end{${target}}`);
    })
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, inner) => display(inner));

  return body;
}

/** Render a (subset of) LaTeX document source to an HTML preview, never throwing. */
export function renderLatex(src: string): string {
  try {
    return marked.parse(texToMarkdown(src)) as string;
  } catch (e) {
    return `<pre class="md-err">preview error: ${String(e)}</pre>`;
  }
}
