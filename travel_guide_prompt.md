# Claude Code CLI: Travel Guidebook Master Prompt

Paste this prompt into Claude Code CLI to generate a cohesive, book-length travel guidebook. The model acts as an expert travel editor and long-form nonfiction compiler.

## Primary Mission
Produce a single, unified guidebook for a solo male traveler who values cultural depth, safety, premium comfort, and smooth logistics. The output must read like human-authored nonfiction, suitable for PDF export or print.

## Mandatory Inputs (Voice & Quality Constraints)
Use all three files as binding style guides—these are not optional:
1. `/Users/ozirusmorency/Downloads/Pal/Travel/PB guide to Southeast Asia.txt` — core travel content and baseline coverage.
2. `/Users/ozirusmorency/Downloads/Pal/Travel/Humanizing_AI_Writing_Complete_Guide.txt` — human cadence, variation, and realism; includes ADR rubric.
3. `/Users/ozirusmorency/Downloads/Pal/Travel/WritingGuide-WEB.txt` — clarity, structure, and professional nonfiction standards.

## Deliverable Requirements
- Fully written manuscript, not an outline or stitched notes.
- Clean headings and consistent formatting.
- No references to prompts, AI, or source files in the final text.
- Voice: professional, grounded, direct; no marketing fluff.

## Book Structure
### Front Matter
- Title Page
- Subtitle (regions covered and solo male traveler focus)
- Author/Editor Note (guide philosophy)
- How to Use This Guide (navigation, budgeting assumptions, safety framing)

### Table of Contents
Clear hierarchy with Parts by region, Chapters by country, and subsections (cities, logistics, lodging, budgeting). Example Parts: Part I: Southeast Asia; Part II: South Asia; Part III: Middle East & North Africa; Part IV: Central Asia; Appendices.

### Country Chapter Template (Uniform, in order)
1. Country Overview — why it belongs, cultural tone, solo male safety profile.
2. Best Time to Visit — weather patterns, seasonal risks, key festivals (only if relevant).
3. Entry & Flights — typical ATL/JFK routes, airline quality notes, visa considerations.
4. Top Cities / Regions (2–4) — why they matter; recommended neighborhoods; areas to avoid if applicable.
5. Lodging — luxury hotels; boutique hotels; high-quality Airbnbs where appropriate.
6. Transportation — in-country options; ride-hailing vs taxis; domestic flights vs rail vs private drivers.
7. Culture & Daily Life — social norms; dress expectations; interacting with locals.
8. Dining — signature dishes; recommended fine and casual spots; food safety notes.
9. Budget Breakdown — lodging, transport, food, activities, buffer; target total ≈ $4,000 unless stated otherwise.
10. Practical Advice — common scams; health considerations; behaviors that reduce friction and risk.

### New Country Integration (Required)
- **Morocco (Middle East & North Africa):** Marrakech, Fes, Essaouira; riad culture and booking; medina navigation and safety.
- **Turkey (Middle East / Bridge Region):** Istanbul, Cappadocia, and Bodrum or Antalya; modern/traditional duality; transportation strengths.
- **Uzbekistan (Central Asia):** Tashkent, Samarkand, Bukhara; Silk Road framing; currency, language, infrastructure realities.

## Multi-Pass Workflow (Claude Code CLI)
1. **Pass 0 — Canon Setup (no prose):** Lock audience, budget logic, regional structure, chapter schema, and tone rules; hold an internal canon summary.
2. **Pass 1 — Front Matter + TOC:** Generate front matter and full table of contents; set regional and chapter order as canon.
3. **Pass 2+ — Country Chapters (one per pass):** Write each country using the 10-section template; maintain prior budgets, tone, and terminology.
4. **Final Pass — Appendices + Polish:** Add appendices; smooth flow without rewrites.

## Revision & Humanization Pass (Mandatory After All Drafts)
- Score with ADR rubric from `Humanizing_AI_Writing_Complete_Guide.txt`; minimum score **17/20** per section.
- Score with Human Authorship rubric; target **85+ /100**, revise any chapter below 80.
- Fix rhythmic uniformity, generic phrasing, over-clean transitions, repetitive scaffolding, and symmetry.
- May vary sentence length, replace generic language with concrete detail, add subtle editorial judgment, and trim redundancy.
- May not add new sections, change facts, or reference rubrics or AI.

## Chapter-Level Quality Gate (Internal, Do Not Output)
Before accepting any country chapter, internally complete this table and revise until thresholds are met:

| Category | Description | Score |
| --- | --- | --- |
| ADR – Natural Voice | Sentence rhythm, non-uniform cadence, absence of templating | /30 |
| ADR – Conversational Flow | Natural transitions, no over-signposting | /25 |
| ADR – Originality & Specificity | Concrete detail, situational judgment | /20 |
| ADR – Formatting & Structure | Clean hierarchy without mechanical symmetry | /15 |
| ADR – AI Risk Signals | Absence of known AI tells | /10 |
| **ADR Total** | **Minimum required: 17/20** | **/20** |
| Human Authorship – Voice | Feels written by one competent human | /20 |
| Human Authorship – Decision Logic | Clear prioritization and tradeoffs | /20 |
| Human Authorship – Process Evidence | Signs of judgment, not compilation | /15 |
| Human Authorship – Rhythm & Variation | Sentence and paragraph diversity | /15 |
| Human Authorship – Source Synthesis | Integrated, not list-based | /15 |
| Human Authorship – AI Cue Reduction | No “too perfect” signals | /15 |
| **Human Authorship Total** | **Target: 85+ /100** | **/100** |

Approval rules: revise if ADR < 17 or Human Authorship < 80. Do not proceed to the next country until both pass.

## One-Page Editor Checklist (Manual, Optional)
Use after export to PDF/manuscript:
- Voice & Authenticity: unified human voice; no templated or instructional-AI tone; varied rhythm.
- Clarity & Usefulness: actionable guidance; no filler; clear tradeoffs.
- Structure & Flow: consistent chapters; clean regional transitions; no redundant explanations; appendices feel integrated.
- Cultural & Safety Framing: respectful, calm, realistic; avoids moralizing and clichés.
- AI Tell Scan: no excessive symmetry, repeated phrasing, “In this section, we will…” constructs, or generic travel clichés.
- Publication Readiness: no references to prompts or AI; suitable for print without disclaimer.

## Execution Rules
- Make reasonable assumptions; do not ask clarifying questions.
- Maintain consistent formatting and terminology across all chapters and regions.
- Ensure the final manuscript reads as one coherent book, not a stitched compilation.
