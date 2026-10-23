from pathlib import Path
import markdown

folder = Path(__file__).resolve().parent
source = folder / "BTP_report_draft.md"
output = folder / "BTP_Report.html"

text = source.read_text(encoding="utf-8")

html_body = markdown.markdown(
    text,
    extensions=["tables", "fenced_code", "toc", "sane_lists"]
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTP Research Report</title>
<style>
@page {{
    size: A4;
    margin: 22mm 20mm 22mm 25mm;
}}
body {{
    font-family: Arial, Helvetica, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #222;
    max-width: 900px;
    margin: 35px auto;
    padding: 0 25px;
}}
h1, h2, h3, h4 {{
    color: #17365D;
    line-height: 1.25;
    page-break-after: avoid;
}}
h1 {{
    font-size: 23pt;
    border-bottom: 2px solid #17365D;
    padding-bottom: 8px;
}}
h2 {{ font-size: 17pt; margin-top: 28px; }}
h3 {{ font-size: 13pt; margin-top: 22px; }}
p {{ text-align: justify; }}
img {{
    display: block;
    max-width: 100%;
    max-height: 165mm;
    height: auto;
    margin: 18px auto;
    page-break-inside: avoid;
    page-break-after: avoid;
    break-inside: avoid-page;
}}
img + p {{
    page-break-before: avoid;
    break-before: avoid-page;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 16px 0;
    font-size: 9pt;
    page-break-inside: auto;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 6px 8px;
    text-align: left;
}}
th {{ background: #eaf0f7; }}
tr {{ page-break-inside: avoid; }}
pre, code {{
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}
blockquote {{
    border-left: 3px solid #8aa4c8;
    margin-left: 0;
    padding-left: 15px;
    color: #555;
}}
a {{ color: #174a7e; overflow-wrap: anywhere; }}
@media print {{
    body {{ max-width: none; margin: 0; padding: 0; }}
    a {{ color: inherit; text-decoration: none; }}
    h1, h2, h3 {{ page-break-after: avoid; }}
}}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

output.write_text(html, encoding="utf-8")
print(f"Created: {output}")
print("Images are expected in the same directory as this HTML file.")
