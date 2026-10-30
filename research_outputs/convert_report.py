import html as html_lib
import re
from pathlib import Path

import markdown

# File paths
folder = Path(__file__).resolve().parent
source = folder / "BTP_report_draft.md"
output = folder / "BTP_Report.html"

# Read the Markdown report
if not source.exists():
    raise FileNotFoundError(f"Report source not found: {source}")

text = source.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Protect mathematics from the Markdown parser.
#
# Python-Markdown treats a backslash before punctuation as an escape, so
# \( ... \) becomes ( ... ) and \[ ... \] becomes [ ... ] before MathJax ever
# sees them. That is why the equations printed as raw LaTeX. The fix is to cut
# every math expression out BEFORE conversion, leave a plain placeholder, and
# put the expression back (HTML-escaped, with canonical delimiters) afterwards.
# Supported source delimiters: \( \), \[ \], $$ $$ and $ $.
# Code spans and fenced code blocks are left untouched.
# ---------------------------------------------------------------------------
CODE_PATTERN = re.compile(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", re.DOTALL)
MATH_PATTERNS = [
    ("display", re.compile(r"\$\$(.+?)\$\$", re.DOTALL)),
    ("display", re.compile(r"\\\[(.+?)\\\]", re.DOTALL)),
    ("inline", re.compile(r"\\\((.+?)\\\)", re.DOTALL)),
    ("inline", re.compile(r"(?<![\\$\w])\$(?=\S)([^$\n]+?)(?<=\S)\$(?![\w$])")),
]
math_store = []


def protect_math(segment):
    for kind, pattern in MATH_PATTERNS:
        def stash(match):
            body = match.group(1)
            if "\n\n" in body:          # never swallow a paragraph break
                return match.group(0)
            math_store.append((kind, body.strip()))
            return f"MJXPH{len(math_store) - 1}MJXPH"
        segment = pattern.sub(stash, segment)
    return segment


parts = CODE_PATTERN.split(text)        # odd indices are code, even are prose
for i in range(0, len(parts), 2):
    parts[i] = protect_math(parts[i])
text = "".join(parts)

# Convert Markdown to HTML
html_body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "toc", "sane_lists"]
)


def restore_math(match):
    kind, body = math_store[int(match.group(1))]
    body = html_lib.escape(body, quote=False)
    return f"\\[{body}\\]" if kind == "display" else f"\\({body}\\)"


html_body = re.sub(r"MJXPH(\d+)MJXPH", restore_math, html_body)
assert "MJXPH" not in html_body, "A math placeholder was not restored"

# ---------------------------------------------------------------------------
# Keep each figure together with its caption (prevents a caption being split
# from its image across a page break).
# ---------------------------------------------------------------------------
html_body, figure_count = re.subn(
    r"<p>\s*(<img [^>]*?/?>)\s*</p>\s*<p>\s*(<strong>Figure\s+\d+\.</strong>.*?)</p>",
    r"<figure>\1<figcaption>\2</figcaption></figure>",
    html_body,
    flags=re.DOTALL,
)

# Warn about images that are referenced but missing next to this file
missing_images = [
    src for src in re.findall(r'<img [^>]*src="([^"]+)"', html_body)
    if not src.startswith(("http://", "https://", "data:")) and not (folder / src).exists()
]

# Generate the complete HTML document
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>BTP Research Report</title>

<!-- MathJax configuration for rendering mathematical equations -->
<script>
MathJax = {{
    tex: {{
        inlineMath: [['$', '$'], [String.raw`\\(`, String.raw`\\)`]],
        displayMath: [['$$', '$$'], [String.raw`\\[`, String.raw`\\]`]]
    }},
    options: {{
        skipHtmlTags: [
            'script', 'noscript', 'style', 'textarea', 'pre', 'code'
        ]
    }}
}};
</script>

<script
    defer
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>

<style>
/* A4 page layout */
@page {{
    size: A4;
    margin: 22mm 20mm 22mm 25mm;
}}

/* Main report typography */
body {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #222;
    max-width: 900px;
    margin: 35px auto;
    padding: 0 25px;
}}

/* Headings */
h1, h2, h3, h4 {{
    line-height: 1.3;
    color: #111;
}}

h1 {{
    font-size: 23pt;
    margin-top: 28px;
    margin-bottom: 16px;
}}

h2 {{
    font-size: 17pt;
    margin-top: 25px;
    margin-bottom: 12px;
}}

h3 {{
    font-size: 13pt;
    margin-top: 20px;
    margin-bottom: 9px;
}}

/* Paragraphs and lists */
p {{
    text-align: justify;
    margin: 9px 0;
}}

li {{
    margin-bottom: 4px;
}}

/* Figures: image and caption are kept together on one page */
figure {{
    margin: 14px 0;
    page-break-inside: avoid;
    break-inside: avoid;
}}

figcaption {{
    text-align: justify;
    font-size: 10pt;
    margin-top: 6px;
}}

img {{
    display: block;
    width: auto;
    max-width: 100%;
    max-height: 185mm;
    height: auto;
    margin: 0 auto;
    page-break-inside: avoid;
    break-inside: avoid;
}}

/* Tables */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9pt;
    page-break-inside: auto;
}}

thead {{
    display: table-header-group;
}}

th, td {{
    border: 1px solid #bbb;
    padding: 6px 8px;
    text-align: left;
    vertical-align: top;
    overflow-wrap: anywhere;
}}

th {{
    background-color: #f0f0f0;
    font-weight: bold;
}}

tr {{
    page-break-inside: avoid;
}}

/* Code and long text */
pre, code {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}

/* Mathematical equations */
.MathJax {{
    font-size: 1em !important;
}}

mjx-container[display="true"] {{
    margin: 14px 0 !important;
    overflow-x: auto;
    overflow-y: hidden;
    page-break-inside: avoid;
    break-inside: avoid;
}}

/* Links */
a {{
    color: #174ea6;
    overflow-wrap: anywhere;
}}

/* Avoid splitting headings from the following content */
h1, h2, h3, h4 {{
    page-break-after: avoid;
    break-after: avoid;
}}

/* Print-specific layout */
@media print {{
    body {{
        max-width: none;
        margin: 0;
        padding: 0;
        font-size: 10.5pt;
    }}

    img {{
        max-width: 100%;
        max-height: 185mm;
    }}

    a {{
        color: inherit;
        text-decoration: none;
    }}

    table {{
        font-size: 8.5pt;
    }}

    h1, h2, h3, h4 {{
        page-break-after: avoid;
        break-after: avoid;
    }}

    p, li {{
        orphans: 3;
        widows: 3;
    }}
}}
</style>
</head>

<body>
{html_body}
</body>
</html>
"""

# Write the generated HTML
output.write_text(html, encoding="utf-8")

print(f"Created: {output}")
print(f"Math expressions protected: {len(math_store)} "
      f"({sum(k == 'display' for k, _ in math_store)} display, "
      f"{sum(k == 'inline' for k, _ in math_store)} inline)")
print(f"Figures kept together with their captions: {figure_count}")
if missing_images:
    print("WARNING: images referenced but not found next to the HTML file:")
    for name in missing_images:
        print(f"  - {name}")
print("MathJax requires an internet connection to render equations. "
      "Open the HTML in a browser and wait for the equations to appear before printing.")