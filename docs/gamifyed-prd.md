# Product Requirements Document (PRD)

## Product Name
**ADHD Gamified Learning System ("GamifyED")**

## Purpose and Scope
GamifyED is a web-based learning and executive-function support system for adults with severe ADHD who need to master dense, expert-level material (e.g., graduate STEM courses, technical certification, jurisprudence, and complex humanities).

The product does **not** claim to cure ADHD or offer photographic memory. Instead, it combines evidence-backed learning mechanics, ADHD-specific support patterns, and purposeful gamification to improve long-term retention and performance consistency. When paired with treatment and/or coaching, GamifyED is intended to help users move from fragmented learning to reliable mastery.

## Target Audience
- Adults (18+) with severe ADHD or related executive-function challenges.
- Learners with high cognitive potential but inconsistent execution.
- Users who want serious, measurable outcomes (not therapy replacement or childish UX).

## Product Goals
1. **Improve Learning Reliability**  
   Increase encoding and retrieval of dense material using retrieval practice, spaced repetition, interleaving, and elaboration.
2. **Support Executive Function**  
   Externalize planning, sequencing, working-memory offloading, task initiation, and interruption re-entry.
3. **Gamify Meaningfully**  
   Use XP, levels, missions, boss fights, skill trees, rank states, and recovery metrics to reinforce competence—not passive exposure.
4. **Enable Recovery and Re-entry**  
   Provide non-shaming, structured re-entry ramps after interruptions or missed sessions.
5. **Integrate with Support Stack**  
   Complement medication/therapy/coaching (when present), without replacing clinical care.

---

## Functional Requirements

### 1) Registration and Profile
- Secure sign-up/login via email/password and social auth (GitHub, Google).
- Collect only minimal required data: name, email, study domains.
- Store and edit goals, domains, time commitment, preferred session length.
- Show progress over time.

### 2) Mission-Based Learning Structure
- Mission = discrete capability unit (e.g., reconstruct theorem proof, explain Bayesian diagnostics).
- Mission includes objectives, prerequisites, retrieval prompts, and boss tests.

#### Mission Board / Home Screen
- Show one active mission.
- Show next micro-step and selectable time box (15/30/60/90 min).
- Show mastery state progression:
  - Fragile
  - Stable
  - Transfer-ready
  - Operational
  - Teach-ready
- Show last failure point and a **Resume from last known state** action.
- Keep decision load intentionally low.

### 3) Structured Sessions
- Modes:
  - 15-minute rescue
  - 30-minute standard
  - 60-minute deep work
  - 90-minute mastery
- Session flow:
  **Briefing → Exposure/Sprint → Closed-book Retrieval → Repair → Failure Logging → Reward → Next Review Scheduling**
- Recovery mode for quick re-entry after interruptions.

### 4) Retrieval Practice and Spaced Repetition
- Scheduler supports spaced reviews (SM-2 or custom variant).
- Interval decisions use recall success, confidence, and delay.
- Prompt types prioritize active recall:
  - Open-ended
  - Cloze
  - Forced production
- Interleave across missions to reduce familiarity illusions.

### 5) Thread Sheets and Artifacts
For each mission, auto-generate and maintain a **Thread Sheet** with:
- definitions
- dependency chain
- canonical example
- failure case
- retrieval prompts

Additional artifacts:
- proof skeletons
- concept maps
- error logs
- decision trees
- session summaries

### 6) Progress Tracking and Gamification
- **XP + Levels** awarded for retrieval success, recovery, and mastery gains.
- **Skill trees** represent dependency graph and bottlenecks.
- **Boss fights** (weekly cumulative transfer checks).
- **Recovery metrics** replace shame-based streak logic.
- Optional forum/leaderboard with moderation and opt-in social sharing.

### 7) Notifications and Reminders
- Push/email reminders for sessions, due reviews, boss fights, and recovery opportunities.
- Fully customizable frequency and opt-out.
- Messaging style: encouraging re-entry, never guilt-inducing.

### 8) Coach/Clinician Integration (Optional)
- Opt-in read-only dashboard for progress and error patterns.
- Privacy-first data sharing and explicit consent.

### 9) Accessibility and Responsiveness
- Responsive desktop/tablet/mobile.
- High-contrast themes and adjustable font size.
- Avoid visual clutter and overstimulating animation.

---

## Non-Functional Requirements
- **Security & Privacy:** HTTPS, bcrypt/argon2, JWT sessions, GDPR/CCPA alignment, clear policy.
- **Performance:** initial load ≤ 2s on broadband; scheduler/mission retrieval response target < 200ms.
- **Scalability:** stateless APIs, caching, horizontal backend scaling.
- **Reliability:** 99.5% uptime, resilient scheduled-task retries, robust error handling.
- **Maintainability:** modular architecture, automated testing, complete docs.

---

## Suggested Technology Stack
- **Frontend:** Next.js + React + TypeScript; Chakra UI or Tailwind; Redux Toolkit or Zustand.
- **Backend:** Node.js with Express or NestJS; REST and/or GraphQL.
- **Database:** PostgreSQL (+ Redis for caching/scheduling).
- **Scheduling:** BullMQ (or equivalent).
- **Auth:** JWT + refresh tokens + OAuth providers.
- **Deployment:** Docker + cloud hosting (AWS ECS / Azure App Service) + CI/CD.

---

## Data Model (High-Level)

| Entity | Purpose | Key Fields |
|---|---|---|
| User | Profile, preferences, auth | id, email, hashed_password, goals, session_preferences, created_at |
| Mission | Unit of learning | id, title, description, prerequisites, retrieval_prompts, canonical_example, failure_example, domain, created_by |
| Session | Learning session record | id, user_id, mission_id, mode, start_time, end_time, retrieval_score, error_log, next_review_date |
| ThreadSheet | Mission memory artifact | id, mission_id, user_id, content, last_updated |
| XPEvent | XP audit trail | id, user_id, mission_id, date, xp_amount, reason |
| SkillTreeNode | Dependency structure | id, mission_id, parent_id, state |
| Notification | Reminder queue | id, user_id, type, scheduled_at, sent_at |

---

## Primary User Journey
1. **Onboarding:** sign up, select domains, set goals, run baseline calibration mission.
2. **Mission Selection:** system recommends next mission from dependencies + goals.
3. **Start Session:** user picks mode (15/30/60/90), receives brief and timer.
4. **Exposure:** chunked content interaction (text, diagrams, code, video).
5. **Retrieval:** active recall prompts (typed/spoken).
6. **Repair:** compare to canonical answer; patch omissions; update thread sheet.
7. **Logging + Reward:** tag error type, award XP, update mastery, schedule follow-up.
8. **Spaced Return:** reminder when due; resume from last failure point.
9. **Progress View:** monitor mastery, XP, recovery performance, upcoming assessments.

---

## MVP Scope (First Release)
Include only features required to validate learning and retention outcomes:
- Core mission engine (objectives, prerequisites, retrieval prompts, canonical examples)
- 15/30-minute session modes and core loop
- Thread sheets + error logs
- Basic spaced repetition scheduler
- XP + levels + mission-board mastery state
- Authentication + profile + goals
- Responsive, uncluttered professional UI (dark/light + high contrast)

Defer to post-MVP:
- boss fights
- full skill tree visualizer
- recovery metrics depth
- forums/leaderboards
- clinician dashboards
- dedicated mobile apps

---

## Validation and Measurement
- **Engagement:** session completion, rescue session frequency, re-entry latency.
- **Learning:** retrieval accuracy, mastery progression (Fragile → Stable → Transfer-ready).
- **Executive Function:** reduced startup friction, fewer abandoned sessions, more successful re-entries.
- **Functional Self-report:** planning confidence and resume ability.
- **Clinical/Coach Feedback (optional):** observed functional gains with support stack.

---

## Out of Scope / Non-Requirements
- Claiming to cure/reverse ADHD.
- Rewarding passive time/page consumption.
- Childish gamification patterns.
- Replacing therapy, medication, or professional care.

---

## Acceptance Criteria
- User can register, select a domain, and start a first mission.
- Session flow (briefing → exposure → retrieval → repair → logging) is understandable and complete.
- Thread sheet updates after session and includes retrieval prompts.
- XP/levels reflect retrieval and recovery (not reading time).
- User receives due-review reminders and can resume from last failure point.
- UI remains professional and uncluttered across desktop/tablet/mobile.
