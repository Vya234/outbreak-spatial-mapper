from pathlib import Path
import re

p = Path("research_outputs")
s = (p / "BTP_report_draft.md").read_text()
imgs = re.findall(r'!\[.*?\]\((.*?)\)', s)

print("Figures found:", len(imgs))
for img in imgs:
    print("OK" if (p / img).exists() else "MISSING", img)

print("Report lines:", len(s.splitlines()))
