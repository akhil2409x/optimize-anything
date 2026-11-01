# AI tool comparison (Section 8)

**Task (identical for both tools):** implement the local CSE pass for the Tiny
compiler's IR. The exact prompt is in `tracks/F_compiler/tool_comparison/PROMPT.md`.

## How to run it (student)

1. Open a **new** chat in tool 1 (e.g. ChatGPT) and paste everything under the
   line in `PROMPT.md`, unchanged. Save the code as
   `tracks/F_compiler/tool_comparison/submission_chatgpt.py`.
2. Do the same in a **new** chat in tool 2 (e.g. Claude, a fresh conversation
   with no project context). Save it as `submission_claude.py`.
3. Note the wall-clock time from sending the prompt to having working code, and
   any follow-up prompts you needed.
4. Score each answer:

   ```bash
   cd tracks/F_compiler
   python tool_comparison/check_submission.py tool_comparison/submission_chatgpt.py
   python tool_comparison/check_submission.py tool_comparison/submission_claude.py
   ```

5. Save screenshots or exports of both chats in `prompts/F/exports/`, then
   fill in the table.

## Results

| Criterion | Tool 1: ______ | Tool 2: ______ |
|---|---|---|
| Model / version | | |
| Hazard unit tests passed (of 10) | | |
| Random programs miscompiled (of 300) | | |
| Crashes | | |
| Hallucinations (invented APIs, false claims in its explanation) | | |
| Instructions executed on 8 benchmarks (in-repo pass: see scorer output) | | |
| Lines of code | | |
| Follow-up prompts needed | | |
| Time to working code | | |
| Which version would you keep, and why | | |

**Scorer sanity check (done in the session):** the in-repo CSE pass scores
10/10 with 0 miscompiled programs. A deliberately naive CSE (no invalidation,
not commutative) scores 5/10 and miscompiles random program 177.

## Notes
