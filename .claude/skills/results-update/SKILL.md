---
name: results-update
description: Use when RESULTS.md (or any results summary, README numbers, or write-up) needs to change. Every number must come from a logged row in data/results.csv or a RESULT section in PREREGISTRATIONS.md; nothing is re-run, re-tuned or recalled from memory.
---

# Update RESULTS.md

1. Collect sources first: the `results.csv` rows (by mode, config, confirmatory flag, time) and the RESULT sections. List them before editing.
2. Each number you write must match a source exactly (same rounding as the source). Put pre-registered (`confirmatory=1`) results in the confirmatory table and everything else under exploratory, labelled as such.
3. Keep CIs, effect directions, unit of analysis and the number of sessions / targets next to each effect. A NOT SUPPORTED result stays in the table.
4. Update caveats if the new result changes what can be claimed (e.g. still one person / one room).
5. Do not re-run analyses to "refresh" numbers. If a number looks wrong, report the discrepancy to the author instead of changing it.
6. Ask the `stats-reviewer` agent to check the edited table against the sources before committing.
