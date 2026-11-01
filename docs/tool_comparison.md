# AI tool comparison (Section 8)

**Task (identical for every tool):** implement the local CSE pass for the Tiny
compiler's IR. The exact prompt is in `tracks/F_compiler/tool_comparison/PROMPT.md`.

## Setup

| | Tool 1 | Tool 2 | Reference |
|---|---|---|---|
| Tool | Claude Sonnet 5.5 | Claude Haiku 5.5 | Claude Opus 5.5 (the main session) |
| How it was run | fresh-context subagent, no tools, no repository access | same | wrote the in-repo pass during Track F, with tests and fixes |
| Date | 2026-11-01 | 2026-11-01 | 2026-10-29 |

- **Identical prompts:** both tools got the prompt text from `PROMPT.md`
  word for word.
- **No outside help:** each was told to answer from its own knowledge,
  without tools or files, so neither could see the repository or the scorer.
- **Saved unchanged:** the replies are saved as
  `tracks/F_compiler/tool_comparison/submission_claude_sonnet.py` and
  `submission_claude_haiku.py`. The only change was decoding HTML entities
  (`&gt;` → `>`).
- **Same check for both:** each was scored by `check_submission.py`.
  - **Unit tests:** 10 targeted tests, one per known hazard.
  - **Differential tests:** 300 random programs, run with CSE alone and
    inside the full pipeline, against an unoptimized interpreter.
  - **Benchmarks:** instructions executed on the 8 benchmark programs.

## Results

| Criterion | Claude Sonnet 5.5 | Claude Haiku 5.5 | In-repo pass (Opus 5.5) |
|---|---|---|---|
| Hazard unit tests passed (of 10) | **10** | **10** | 10 |
| Random programs miscompiled (of 300) | 0 | 0 | 0 |
| Crashes | 0 | 0 | 0 |
| Instructions executed on 8 benchmarks | **281,982** | **281,978** | 287,385 |
| Lines of code | 72 | 85 | 22 |
| Hallucinations | none found; one unused import (`dataclass`, `field`) | none found | — |
| Follow-up prompts needed | 0 | 0 | several during Track F (see prompts/F/001) |
| Time to answer | 23 s | 54 s | — |
| Scoring time | 4.2 s | 5.8 s | — |

## Discussion

- **Correctness:** both tools produced correct code on the first try. Each
  handles the hazards the prompt hides:
  - a variable redefined after an expression is recorded;
  - commutative operands;
  - division by zero (the first division raises, so reusing its result is
    safe);
  - `x = x + 1`, where the destination is also an operand.
- **The tools beat the reference by ~1.9%:** both did real *value*
  numbering (a value number per variable, followed through copies). The
  in-repo pass is simpler. It keys expressions on variable *names* and
  invalidates on redefinition, so it misses `b = a; c = a + 1; d = b + 1`.
- **Readability:** the in-repo pass is shortest (22 lines). Haiku's is the
  most explicit (separate helper, comments).
  - Haiku also skips an instruction whose destination already holds the
    value.
  - Sonnet re-uses a value number when no variable still holds it.
  - Both of those choices are correct.
- **Speed and iterations:** Sonnet answered in less than half Haiku's time.
  Neither needed a second prompt.
- **Hallucinations:** none in the code. Neither reply contained claims to
  check, because both followed "reply with just the code".

## Caveats

- **Same vendor:** both tools are Claude models. The brief says "at least two
  AI tools". This compares two different models under identical conditions,
  but a comparison across vendors would be stronger.
- **Adding a non-Anthropic tool:** Codex and Gemini appear to be installed on
  the student's computer, but this session could only reach a Linux VM on
  it, not the macOS shell. To add one, run (from `tracks/F_compiler`):

  ```bash
  sed '1,/^---$/d' tool_comparison/PROMPT.md > /tmp/prompt.txt
  codex exec "$(cat /tmp/prompt.txt)" > tool_comparison/submission_codex.md   # or: gemini -p "$(cat /tmp/prompt.txt)"
  # copy the code block into tool_comparison/submission_codex.py, then:
  python tool_comparison/check_submission.py tool_comparison/submission_codex.py
  ```

  Then add a column above and register the tool in the root README.
- **Single run:** one sample per model. Sampling again could give different
  code.
- **Easier than real use:** this task has a precise spec and a strong
  scorer. Real use, with vaguer prompts, would show bigger differences.
