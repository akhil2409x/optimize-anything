"""Convert a Claude session transcript (JSONL) into a readable Markdown chat export.

User messages, assistant text and every tool call are kept; long tool inputs and
outputs are truncated. Secrets are scrubbed (tokens, keys, e-mail addresses)."""
import json
import re
import sys
from pathlib import Path

SECRET = [(re.compile(r"(gh[pousr]_[A-Za-z0-9]{20,})"), "<redacted-token>"),
          (re.compile(r"(sk-[A-Za-z0-9_-]{20,})"), "<redacted-key>"),
          (re.compile(r"((?:GH_TOKEN|GITHUB_TOKEN|API_KEY|PASSWORD)\s*[=:]\s*)\S+", re.I), r"\1<redacted>"),
          (re.compile(r"[A-Za-z0-9._%+-]+@(?!anthropic\.com)[A-Za-z0-9.-]+\.[a-z]{2,}"), "<email>"),
          (re.compile(r"<system-reminder>.*?</system-reminder>", re.S), "")]


def scrub(s):
    for rx, rep in SECRET:
        s = rx.sub(rep, s)
    return s


def cut(s, n):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + f"\n… [{len(s) - n} more characters truncated]"


def render(path, title, user_label="Student"):
    out = [f"# {title}", "", f"Source: `{Path(path).name}` (exported by `tools/export_transcript.py`; "
           "tool inputs truncated to 1,500 and outputs to 800 characters; secrets and e-mail addresses redacted).", ""]
    for line in open(path):
        d = json.loads(line)
        t = d.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = d.get("message", {})
        content = msg.get("content")
        ts = d.get("timestamp", "")[:19].replace("T", " ")
        items = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
        for c in items:
            k = c.get("type")
            if k == "text" and c.get("text", "").strip():
                txt = scrub(c["text"]).strip()
                if not txt:
                    continue
                who = f"**{user_label}**" if t == "user" else "**Claude**"
                if t == "user" and (txt.startswith("[SYSTEM NOTIFICATION") or txt.startswith("<task-notification")
                                    or "This session is being continued from a previous conversation" in txt):
                    who = "*System / harness*"
                out += [f"### {who} · {ts}", "", txt, ""]
            elif k == "tool_use":
                inp = c.get("input", {})
                desc = inp.get("description", "")
                body = inp.get("command") or inp.get("content") or inp.get("prompt") or inp
                out += [f"<details><summary>Tool call: {c.get('name')} — {scrub(desc)}</summary>", "",
                        "```", scrub(cut(body, 1500)), "```", "</details>", ""]
            elif k == "tool_result":
                r = c.get("content")
                if isinstance(r, list):
                    r = "\n".join(x.get("text", "[image]") for x in r if isinstance(x, dict))
                out += ["<details><summary>Tool result</summary>", "", "```", scrub(cut(r or "", 800)), "```",
                        "</details>", ""]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    src, dst, title = sys.argv[1], sys.argv[2], sys.argv[3]
    label = sys.argv[4] if len(sys.argv) > 4 else "Student"
    Path(dst).write_text(render(src, title, label))
    print(dst, Path(dst).stat().st_size)
