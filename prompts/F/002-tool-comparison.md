# F · 002 · AI tool comparison run

- **Tool / model:** Claude Opus 5.5 (main session) ran Claude Sonnet 5.5 and
  Claude Haiku 5.5 as fresh-context subagents.
- **Date:** 2026-11-01

## Original prompt

```
do it all for me
```

## What was done

**The prompt:**
- The text of `tracks/F_compiler/tool_comparison/PROMPT.md` was sent word
  for word to each model.
- Each model was told to answer from its own knowledge, with no tools and no
  file access.

**Saving the answers:**
- Each reply was saved unchanged (HTML entities decoded) as
  `submission_claude_sonnet.py` and `submission_claude_haiku.py`.
- Full transcripts: `exports/tool-comparison-*.md`.

**Scoring:** `check_submission.py` scored both 10/10 hazard tests, with 0 of
300 random programs miscompiled.

**A claim checked by hand:** "both beat the in-repo pass by ~1.9%".
`b = a; c = a + 1; d = b + 1` is reduced by Sonnet's pass but not by the
in-repo one.

**Limitation:** both tools are Claude models. The commands to add Codex or
Gemini are in `docs/tool_comparison.md`.

## Verification note

To do by the student: read both submissions and explain how each one
handles a variable redefined after an expression was recorded.
