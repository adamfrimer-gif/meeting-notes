#!/usr/bin/env python3
"""Render meeting notes Markdown to a self-contained HTML and an A4 PDF.

Images referenced as ![alt](path) are embedded as base64, and an italic line
right after an image becomes its caption. [mm:ss] timestamps become tags.
The PDF is printed with headless Google Chrome.

usage: mnotes pdf notes_vNN.md   (or: .venv/bin/python scripts/notes_to_pdf.py notes_vNN.md)
                  -> notes_vNN.html + notes_vNN.pdf
"""
import base64, html, io, os, re, shutil, subprocess, sys
from PIL import Image

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
]


def find_chrome():
    """CHROME env var (or .env), then the usual macOS app locations, then PATH."""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from envload import load_env
        load_env()
    except Exception:
        pass
    cands = [os.environ.get("CHROME")] + CHROME_CANDIDATES + [
        shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser")]
    for c in cands:
        if c and os.path.exists(os.path.expanduser(c)):
            return os.path.expanduser(c)
    for base in ("/Applications", os.path.expanduser("~/Applications")):
        p = os.path.join(base, "Google Chrome.app/Contents/MacOS/Google Chrome")
        if os.path.exists(p):
            return p
    return None

CSS = """
:root{--ink:#141414;--mute:#5f6368;--line:#e6e6e6;--bg:#fafafa;--accent:#2f6fed;--tint:#eef3fd}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Inter","Segoe UI",Arial,sans-serif}
nav{position:sticky;top:0;z-index:5;background:rgba(20,20,20,.96);padding:10px 24px;display:flex;gap:18px;flex-wrap:wrap;font-size:13px}
nav a{color:#ddd;text-decoration:none}nav a:hover{color:#fff}
main{max-width:860px;margin:0 auto;padding:40px 24px 80px}
h1{font-size:34px;line-height:1.15;margin:0 0 16px;letter-spacing:-.01em}
h2{font-size:22px;margin:48px 0 12px;padding-top:12px;border-top:2px solid var(--ink);scroll-margin-top:60px}
h3{font-size:17px;margin:28px 0 6px}
p,li{margin:6px 0}ul,ol{padding-left:22px}a{color:var(--accent)}
.ts{font:12px/1 ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--tint);color:#1a4fbf;padding:2px 5px;border-radius:4px;white-space:nowrap}
#tldr + p{background:#fff;border-left:4px solid var(--ink);padding:16px 20px;font-size:17px;border-radius:0 6px 6px 0}
figure{margin:18px 0 26px}figure img{width:100%;border:1px solid var(--line);border-radius:8px;display:block;cursor:zoom-in}
figcaption{font-size:13px;color:var(--mute);margin-top:6px}
table{border-collapse:collapse;width:100%;margin:12px 0;background:#fff;font-size:14px}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}th{background:#f1f1f1}
.lb{position:fixed;inset:0;background:rgba(0,0,0,.88);display:none;align-items:center;justify-content:center;z-index:10;cursor:zoom-out}
.lb img{max-width:96vw;max-height:94vh}
@page{size:A4;margin:16mm 16mm 18mm}
@media print{html,body{background:#fff}nav,.lb{display:none}main{max-width:none;padding:0}
h1{font-size:26px}h2{font-size:18px;margin:26px 0 8px;break-after:avoid}h3{font-size:14.5px;margin:16px 0 4px;break-after:avoid}
body{font-size:11.5px;line-height:1.5}
figure{break-inside:avoid;margin:10px 0 14px;text-align:center}figure img{max-height:78mm;width:auto;max-width:100%;margin:0 auto}
figcaption{text-align:left;font-size:10.5px}
#tldr + p{font-size:12.5px;background:#f6f6f6}
.ts{font-size:9.5px;padding:1px 4px;-webkit-print-color-adjust:exact;print-color-adjust:exact}
table{font-size:11px;break-inside:avoid}}
"""

JS = """<div class="lb" id="lb"><img id="lbi"></div><script>
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{lbi.src=i.src;lb.style.display='flex'});lb.onclick=()=>lb.style.display='none';
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.style.display='none'});</script>"""


def data_uri(path):
    im = Image.open(path).convert("RGB")
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2" target="_blank">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[(\d\d:\d\d(?::\d\d)?)\]", r'<span class="ts">\1</span>', s)
    return s


def render(md_path):
    base = os.path.dirname(os.path.abspath(md_path))
    lines = open(md_path).read().split("\n")
    out, toc, lst, i = [], [], [None], 0

    def close():
        if lst[0]:
            out.append("</%s>" % lst[0])
            lst[0] = None

    while i < len(lines):
        l = lines[i]
        if l.startswith("# "):
            close(); out.append("<h1>%s</h1>" % inline(l[2:]))
        elif l.startswith("## "):
            close()
            sid = re.sub(r"\W+", "-", l[3:].lower()).strip("-")
            toc.append((sid, l[3:]))
            out.append('<h2 id="%s">%s</h2>' % (sid, inline(l[3:])))
        elif l.startswith("### "):
            close(); out.append("<h3>%s</h3>" % inline(l[4:]))
        elif l.strip() == "---" or not l.strip():
            close()
        elif l.startswith("!["):
            close()
            alt = re.search(r"!\[([^\]]*)\]", l).group(1)
            src = os.path.join(base, re.search(r"\((.+)\)", l).group(1))
            cap = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("*"):
                cap = inline(lines[i + 1].strip("*")); i += 1
            out.append('<figure><img src="%s" alt="%s"><figcaption>%s</figcaption></figure>'
                       % (data_uri(src), html.escape(alt), cap))
        elif l.startswith("|"):
            close(); rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-| :]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            t = "<table><thead><tr>" + "".join("<th>%s</th>" % inline(c) for c in rows[0]) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join("<td>%s</td>" % inline(c) for c in r) + "</tr>" for r in rows[1:])
            out.append(t + "</tbody></table>")
            continue
        elif re.match(r"^\d+\. ", l):
            if lst[0] != "ol": close(); out.append("<ol>"); lst[0] = "ol"
            out.append("<li>%s</li>" % inline(re.sub(r"^\d+\. ", "", l)))
        elif l.startswith("- "):
            if lst[0] != "ul": close(); out.append("<ul>"); lst[0] = "ul"
            out.append("<li>%s</li>" % inline(l[2:]))
        else:
            close(); out.append("<p>%s</p>" % inline(l))
        i += 1
    close()
    title = next((l[2:] for l in lines if l.startswith("# ")), "Meeting notes")
    nav = "".join('<a href="#%s">%s</a>' % (a, html.escape(b)) for a, b in toc)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            "<title>%s</title><style>%s</style></head><body><nav>%s</nav><main>%s</main>%s</body></html>"
            % (html.escape(title), CSS, nav, "".join(out), JS))


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    md = sys.argv[1]
    stem = os.path.splitext(os.path.abspath(md))[0]
    html_path, pdf_path = stem + ".html", stem + ".pdf"
    for p in (html_path, pdf_path):
        if os.path.exists(p):
            sys.exit("refusing to overwrite %s; bump the _vNN of the .md first" % p)
    chrome = find_chrome()
    if not chrome:
        sys.exit("no Chrome/Chromium found for PDF export; install Google Chrome "
                 "or set CHROME=/path/to/chrome in the skill's .env")
    open(html_path, "w").write(render(md))
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    "--print-to-pdf=" + pdf_path, "file://" + html_path],
                   check=True, capture_output=True)
    print("wrote", html_path)
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
