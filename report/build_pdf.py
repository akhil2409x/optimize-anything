"""Build report/REPORT.pdf from report/REPORT.md (pandoc → HTML → Chromium PDF).

    python report/build_pdf.py        # needs pandoc and playwright (Chromium)
"""
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
body { font-family: 'DejaVu Serif', Georgia, serif; font-size: 10pt; line-height: 1.45; color: #1d2433; max-width: none; }
h1 { font-size: 17pt; color: #14213D; border-bottom: 2px solid #E07A2E; padding-bottom: 3pt; margin-top: 18pt; page-break-after: avoid; }
h2 { font-size: 13pt; color: #14213D; margin-top: 12pt; page-break-after: avoid; }
header h1.title { font-size: 24pt; border: none; margin-bottom: 2pt; }
header .subtitle { font-size: 12pt; color: #5B6676; margin: 0; }
header .author, header .date { font-size: 10pt; color: #5B6676; margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 8.3pt; page-break-inside: avoid; }
th { background: #14213D; color: #F7F5EF; text-align: left; padding: 3pt 5pt; }
td { border-bottom: 1px solid #d9dde4; padding: 3pt 5pt; vertical-align: top; }
tr:nth-child(even) td { background: #f6f4ee; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; background: #f1efe8; padding: 0 2pt; }
hr { border: none; border-top: 1px solid #d9dde4; margin: 14pt 0; }
li { margin: 2pt 0; }
"""


def main():
    html = HERE / "REPORT.html"
    css = HERE / "report.css"
    css.write_text(CSS)
    subprocess.run(["pandoc", str(HERE / "REPORT.md"), "-s", "--toc", "--toc-depth=1", "-c", "report.css",
                    "--metadata", "lang=en", "-o", str(html)], check=True)
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page()
        page.goto(html.resolve().as_uri())
        page.pdf(path=str(HERE / "REPORT.pdf"), format="A4", print_background=True,
                 display_header_footer=True, header_template="<span></span>",
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#888">'
                                 '<span class="pageNumber"></span> / <span class="totalPages"></span></div>',
                 margin=dict(top="16mm", bottom="18mm", left="15mm", right="15mm"))
        b.close()
    html.unlink()
    css.unlink()
    print("wrote", HERE / "REPORT.pdf")


if __name__ == "__main__":
    main()
