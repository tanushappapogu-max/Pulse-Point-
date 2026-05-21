# Contributing to Pulse Point

## Branching

- `main` — stable, deployable. Direct commits go here only for trivial fixes.
- `feat/<short-name>` — new features (e.g., `feat/native-detection`)
- `fix/<short-name>` — bug fixes (e.g., `fix/tracker-stale`)
- `chore/<short-name>` — tooling, deps, docs (e.g., `chore/pin-tfjs`)

Create a branch, make changes, open a pull request against `main`.

## Commit messages

Follow the conventional commit format: `<type>: <short imperative summary>`

| Type | When to use |
|------|-------------|
| `feat` | New user-visible feature |
| `fix` | Bug fix |
| `perf` | Performance improvement |
| `refactor` | Code change with no behavior change |
| `test` | Adding or updating tests |
| `docs` | Documentation only |
| `chore` | Build scripts, dependencies, CI |

Keep the summary under 72 characters. Add a body paragraph if the WHY is not obvious.

Examples:
```
feat: add magnetometer-assisted heading to mobile guidance
fix: clamp SMOOTH_ALPHA to prevent NaN on zero-width bbox
perf: skip NMS when fewer than 5 raw detections
test: cover BoxTracker stale/fresh lifecycle
```

## Running tests

```bash
cd pulse-point
npm install
npm test          # single run
npm run test:watch  # watch mode during development
```

All tests must pass before a PR can be merged.

## Sub-project structure

| Directory | What lives here |
|-----------|-----------------|
| `pulse-point/` | Vite web app — React + TF.js |
| `pulse-point-mobile/` | Expo mobile app — React Native |
| `api/` | Vercel serverless functions |

Keep changes scoped to one sub-project per commit where possible.

## Secrets

Never commit API keys, `.env` files, or credentials. The `.gitignore` already excludes `.env*`. The `OPENROUTER_API_KEY` lives only in Vercel environment variables.
