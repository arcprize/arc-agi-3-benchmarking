# Engineering Coding Standards

**Scope:** These standards are language-agnostic and apply to all teams and repositories in the organization. Per-language style rules (indentation, formatting details, idioms) are handled by each team's linter/formatter config, which should conform to the principles below. Where a language-specific note is useful, it's called out inline.

**Status:** Living document. Propose changes via PR to this file; see [Exceptions & Amendments](#exceptions--amendments).

---

## 1. Purpose

Consistent, readable, well-tested code that any engineer in the org can pick up, review, and safely modify — regardless of which team wrote it. These standards optimize for:

- **Readability over cleverness** — code is read far more often than it's written
- **Consistency over personal preference** — reduces cognitive load across teams
- **Safety over speed** — especially for anything touching production data or user-facing systems

---

## 2. Naming Conventions

- Names should describe **what**, not **how**. `activeUsers` not `filteredList2`.
- Avoid abbreviations except well-known ones (`id`, `url`, `http`). Spell out domain terms in full.
- Booleans read as questions or states: `isActive`, `hasPermission`, `canRetry` — not `active`, `flag`, `status`.
- Functions/methods are verbs or verb phrases: `calculateTotal()`, `sendInvite()`.
- Avoid negated booleans (`isNotValid`) — they compound confusingly with `!`.
- Match the casing convention of the language/ecosystem you're in (e.g., `camelCase` in JS/Java, `snake_case` in Python/Ruby) rather than importing habits from another language.
- Constants representing fixed, meaningful values are named and centralized, not left as unexplained magic numbers/strings scattered through code.

## 3. Code Structure & Organization

- One clear responsibility per function/module. If you need "and" to describe what something does, consider splitting it.
- Prefer flat, shallow logic. More than 3 levels of nesting is a signal to extract a function or invert a condition.
- Group by feature/domain rather than by technical layer when the codebase is large enough to benefit (e.g., `billing/` containing its own models, handlers, tests — not one giant `models/`, `controllers/` split across the whole app).
- Keep public interfaces (exported functions, API contracts, module boundaries) intentionally small. Internal helpers stay private/unexported.
- Delete dead code rather than commenting it out. Version control remembers it for you.

## 4. Version Control (Git)

- **Commits** are small, atomic, and buildable/passing on their own. One logical change per commit.
- **Commit messages** use imperative mood ("Add retry logic", not "Added" or "Adds") with a short summary line (≤72 chars) and, when useful, a body explaining *why*, not just *what*.
- **Branch naming**: `<type>/<short-description>` — e.g., `fix/checkout-timeout`, `feat/sso-login`.
- **Never commit**: secrets, credentials, `.env` files, large binaries, or commented-out blocks of old code.
- **Pull requests** are scoped to one purpose. A PR that mixes a refactor with a feature is harder to review and harder to revert. If it's growing, split it.
- **Rebase vs. merge**: keep history readable — squash noisy WIP commits before merging; avoid rewriting history on shared branches.

## 5. Code Review

- **Every change** to a shared codebase goes through review before merging — no exceptions for "small fixes," since small fixes are exactly where regressions hide.
- **Author responsibilities**: keep PRs small (rule of thumb: <400 lines changed, excluding generated/vendored files), write a description that explains *why* the change exists and how to test it, and self-review the diff before requesting review.
- **Reviewer responsibilities**: respond within one business day; review for correctness, readability, and test coverage — not just style (a linter should catch style, not a human); distinguish blocking comments from suggestions (prefix nits with "Nit:" so authors know what's optional).
- **Disagreements**: if author and reviewer don't converge after 2 rounds, escalate to a tech lead rather than debating in the thread indefinitely.
- At least **one approval** required to merge; two for changes to shared infrastructure, security-sensitive code, or public API contracts.

## 6. Documentation

- **Code comments** explain *why*, not *what* — the code already says what it does. Reserve comments for non-obvious rationale, trade-offs, or warnings ("this looks redundant but is required because X").
- **Public functions/APIs** get a short docstring/comment describing purpose, parameters, return value, and any exceptions/error states — enough that a caller doesn't need to read the implementation.
- **READMEs** for each service/repo cover: what it does, how to run it locally, how to run tests, and how to deploy — kept current as part of the PR that changes any of those things.
- **Architectural decisions** with long-term consequences (choice of database, major dependency, breaking API change) are recorded as a short ADR (Architecture Decision Record): context, decision, consequences.

## 7. Testing

- New logic ships with tests. Bug fixes include a test that would have caught the bug.
- **Test pyramid**: favor many fast unit tests, fewer integration tests, and a small number of end-to-end tests — not the inverse.
- Tests are independent and deterministic — no reliance on execution order, real network calls, or wall-clock time without mocking.
- Test names describe the scenario and expected outcome: `returns_404_when_user_not_found`, not `test1`.
- Coverage targets are a guardrail, not a goal — 100% coverage with meaningless assertions is worse than 80% coverage of real behavior.

## 8. Error Handling & Logging

- Fail loudly in development, gracefully in production. Never silently swallow exceptions (`catch {}` with nothing in it).
- Errors include enough context to debug without reproducing locally: what operation failed, relevant IDs, and the underlying cause.
- Distinguish expected failures (validation errors, not-found) from unexpected ones (bugs, outages) — they should be logged and handled differently.
- Never log secrets, tokens, passwords, or full PII payloads.
- Use structured logging (key-value fields) over free-text string concatenation, so logs are queryable.

## 9. Security

- Never hardcode credentials, API keys, or secrets in source — use the org's secrets manager/environment config.
- Validate and sanitize all external input (user input, API responses, file uploads) before use.
- Use parameterized queries / ORM methods — never string-concatenated SQL.
- Keep dependencies patched; treat a flagged critical/high CVE in a direct dependency as a priority fix, not backlog.
- Principle of least privilege for service accounts, API scopes, and database roles.

## 10. Dependencies

- Prefer the standard library or an existing org-approved dependency before adding a new third-party package.
- Pin dependency versions (lockfiles committed) for reproducible builds.
- New dependencies are a small design decision — check maintenance activity, license compatibility, and bundle/footprint impact before adding one, especially for anything pulled into a shared library.

## 11. CI/CD & Automation

- Formatting and linting are enforced by tooling (e.g., Prettier/Black/gofmt + ESLint/Ruff/golangci-lint), not by review comments. If it can be automated, automate it — don't spend human review time on things a machine catches.
- CI must pass (build, lint, tests) before merge; no merging on red.
- Main/trunk branch is always deployable.

## 12. Language-Specific Notes (brief)

These are pointers, not full style guides — each language should have its own linter config enforcing details.

| Language | Formatter/Linter | Notable convention |
|---|---|---|
| JavaScript/TypeScript | Prettier + ESLint | Prefer `const`; strict null checks in TS; avoid `any` |
| Python | Black + Ruff (or Flake8) | Type hints on public functions; follow PEP 8 |
| Go | `gofmt` + `go vet` | Handle every returned `error` explicitly |
| Java | Checkstyle/Spotless | Favor immutability; explicit `Optional` over null |
| Ruby | RuboCop | Prefer idiomatic blocks over manual loops |

## 13. Enforcement & Tooling

- Standards are enforced primarily through **automation** (linters, formatters, CI checks, pre-commit hooks) and secondarily through **code review** for the judgment calls automation can't make.
- New repos are bootstrapped from the org's shared linter/CI templates rather than reinventing config per team.

## 14. Exceptions & Amendments

- These are defaults, not laws. A team can deviate for a good documented reason — note it in the repo's README or a linter override with a comment explaining why.
- To change this document: open a PR against it, tag affected team leads, allow at least 3 business days for comment before merging.

---
*Last updated: August 2026*
