# BackPack / GamifyED MVP

This repository now includes a working **GamifyED MVP application** based on the PRD.

- PRD: `docs/gamifyed-prd.md`
- App entrypoint: `src/server.js`
- Frontend: `public/index.html`

## Features implemented (MVP)

- Email/password registration and login with JWT auth
- Mission board with mastery-state tracking
- Session lifecycle (start, complete, repair/error logging)
- Thread sheet updates per mission
- Spaced review scheduling (SM-2-inspired interval logic)
- XP events, level calculation, and recovery/re-entry metrics
- Due-review endpoint and dashboard endpoint

## Quick start

```bash
npm install
npm start
```

Open `http://localhost:3000`.

## Smoke test

Run the end-to-end smoke checks (deliverable existence, syntax checks, unit tests, and runtime health check when dependencies can be installed):

```bash
./scripts/smoke-test.sh
```

## API overview

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/profile`
- `PUT /api/profile`
- `GET /api/missions`
- `GET /api/missions/:id`
- `POST /api/sessions/start`
- `POST /api/sessions/:id/complete`
- `GET /api/reviews/due`
- `GET /api/dashboard`
- `GET /api/health`
