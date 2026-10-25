# Agent workflow rules

Rules for every agent working in this repo (planners and executors). Read once, before your
first commit. If this file and any other instruction disagree, this file wins.

## One issue, one branch, one PR

- Work only from the issue you were given. Branch: `feat/<short-slug>` or `fix/<short-slug>`.
- PR body starts with `Closes #N`. Fill in EVERY section of `pull_request_template.md` —
  CI fails on empty sections. The PR body is the permanent record; no separate retro comment.
- Paste real command output in **Verification**. Never write "should pass" or "tested locally".

## Verify before you claim done

- Run the exact commands from the issue's "Verify with" section.
- Tests fail or lint fails → fix it, push to the same branch. Do not open a new PR.

## Never merge

- You open the PR; the human merges it. Never run `gh pr merge`, never click merge, never
  approve your own PR — even if checks pass and it looks ready.

## Stop conditions

- Blocked, ambiguous, or the issue asks for a product decision → stop, comment on the PR or
  return to the caller with the exact question. Do not guess, do not widen scope.
- Never touch anything outside the issue's scope, and never edit `.github/AGENT-WORKFLOW.md`,
  `.github/pull_request_template.md`, or issue templates without an explicit ask.
