# NEXUS — Ethics & Responsible Use

## What NEXUS is

A local-first autonomy harness: it executes user-given objectives with
sandboxed tools and records an auditable trail of every action. Its central
safety property is **verification over trust**: the system is designed around
checking the world instead of believing the model.

## Safeguards built in

1. **Workspace sandbox.** All file tools resolve paths inside one workspace
   directory; traversal attempts raise, never write outside. Unit-tested.
2. **Shell blocklist + timeout.** Destructive commands (`rm -rf /`, `mkfs`,
   `format`, fork bombs, `curl | sh`) are refused; anything else gets 60s.
3. **Step budget.** `max_steps` bounds every run — no infinite loops, no
   runaway token spend.
4. **Full audit trail.** Every objective, assistant turn, tool call and result
   is persisted to SQLite before the next step. Runs are inspectable and
   replayable (`nexus history`, episode events).
5. **No credentials in code.** 12-factor env config; `.env` gitignored.

## Honest limitations

- **The verifier is an LLM.** It can be wrong in both directions (observed in
  testing: rubber-stamping a false success). That is why E2E and any serious
  use should assert ground truth where it exists; the verifier is a critic,
  not the source of truth.
- **Prompt injection is a real risk.** `web_fetch` content is untrusted input;
  an objective that fetches attacker-controlled pages could be steered. The
  sandbox (workspace-only fs, blocked sh) bounds the blast radius to the
  workspace directory — but do not point NEXUS at hostile pages with
  `sh` enabled until a per-run approval gate is added.
- **No spending, messaging, or irreversible actions in the MVP tool set** —
  deliberate. Power tools (browser, external APIs) raise the stakes and need
  an approval gate first.

## Intended use / non-use

- ✅ Personal automation on your own machine, learning, research, portfolio.
- ❌ Not a penetration-testing tool, not for operating on other people's
  systems, not for unattended actions with side effects beyond the workspace.

## Scaling up responsibly

Every capability addition should preserve the invariant: **the agent's reach
must never exceed what its sandbox and audit trail can explain.**
