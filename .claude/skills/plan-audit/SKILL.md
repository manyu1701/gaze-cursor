---
name: plan-audit
description: Use at the start of a working session, before closing a milestone, or when the author asks whether work follows the plan. Compares the repository and recent commits against PLAN.md and reports drift; it changes nothing on its own.
---

# Audit work against PLAN.md

Read PLAN.md, CLAUDE.md, the last ~30 commits (`git log --oneline -30`), RESULTS.md, the PREREGISTRATIONS.md section headings and the status column of `docs/research/ACTIONS.md` (flag R-actions marked adopted that PLAN.md does not mention, and PLAN steps that started without their R-actions being considered). Then report, as a short table plus a recommendation:

1. **Current milestone and step**, and whether its done-when is met (with evidence).
2. **Open items with no home:** steps marked open / carried forward / stretch that no milestone schedules. Propose where each goes, or that it is explicitly dropped, with a reason.
3. **Ordering slips:** work done before a prerequisite the plan names (e.g. a report the plan says must exist before recordings).
4. **Building ahead:** commits that implement a later step without agreement.
5. **Stale text:** statements in PLAN.md / CLAUDE.md contradicted by later results or decisions.
6. **Active pre-registrations:** which are frozen and waiting for data, what data is still needed, and any risk to them.
7. **Process risks:** parallel sessions or diverged branches, uncommitted work, tests failing.
8. **The guard itself** (before closing a milestone): run `python3 .claude/hooks/guard.py doctor`; list the overrides logged in `.claude/state/review_skips.log` and `gate_overrides.log` since the last milestone; propose dropping a rule that only caused friction, or adding one where a real slip got through.

Do not edit PLAN.md during the audit. Present the findings and proposed diff, and apply it only after the author agrees.
