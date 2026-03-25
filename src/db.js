const Database = require('better-sqlite3');
const path = require('path');

const db = new Database(path.join(__dirname, '..', 'gamifyed.db'));
db.pragma('journal_mode = WAL');

db.exec(`
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name TEXT NOT NULL,
  goals TEXT DEFAULT '',
  domains TEXT DEFAULT '',
  weekly_minutes INTEGER DEFAULT 300,
  preferred_session_length INTEGER DEFAULT 30,
  created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS missions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  domain TEXT NOT NULL,
  prerequisites TEXT DEFAULT '[]',
  retrieval_prompts TEXT DEFAULT '[]',
  canonical_example TEXT DEFAULT '',
  failure_example TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  mission_id INTEGER NOT NULL,
  mode TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active',
  retrieval_score REAL DEFAULT 0,
  confidence REAL DEFAULT 0,
  error_log TEXT DEFAULT '',
  started_at TEXT NOT NULL DEFAULT (datetime('now')),
  completed_at TEXT,
  next_review_date TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id),
  FOREIGN KEY(mission_id) REFERENCES missions(id)
);

CREATE TABLE IF NOT EXISTS thread_sheets (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  mission_id INTEGER NOT NULL,
  content TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL DEFAULT (datetime('now')),
  UNIQUE(user_id, mission_id),
  FOREIGN KEY(user_id) REFERENCES users(id),
  FOREIGN KEY(mission_id) REFERENCES missions(id)
);

CREATE TABLE IF NOT EXISTS xp_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  mission_id INTEGER,
  xp_amount INTEGER NOT NULL,
  reason TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (datetime('now')),
  FOREIGN KEY(user_id) REFERENCES users(id)
);
`);

const missionCount = db.prepare('SELECT COUNT(*) as count FROM missions').get().count;
if (missionCount === 0) {
  const insertMission = db.prepare(`
    INSERT INTO missions(title, description, domain, prerequisites, retrieval_prompts, canonical_example, failure_example)
    VALUES(@title, @description, @domain, @prerequisites, @retrieval_prompts, @canonical_example, @failure_example)
  `);

  const missions = [
    {
      title: 'Bayes in Diagnostic Reasoning',
      description: 'Explain Bayes theorem and apply it to medical diagnostic tests.',
      domain: 'statistics',
      prerequisites: '[]',
      retrieval_prompts: JSON.stringify([
        'State Bayes theorem from memory.',
        'Given prevalence, sensitivity, and specificity, compute posterior probability.',
        'Explain base-rate neglect in one paragraph.'
      ]),
      canonical_example: 'P(D|+) = (sens * prev) / ((sens * prev) + ((1-spec) * (1-prev)))',
      failure_example: 'Confusing P(+|D) with P(D|+) and ignoring prevalence.'
    },
    {
      title: 'Diagonalization Proof Reconstruction',
      description: 'Reconstruct Cantor diagonalization and identify failure points.',
      domain: 'mathematics',
      prerequisites: '[]',
      retrieval_prompts: JSON.stringify([
        'List assumptions for contradiction.',
        'Construct diagonal element and explain why it differs.',
        'State final contradiction clearly.'
      ]),
      canonical_example: 'Construct x where x_i differs from i-th list entry at i-th digit.',
      failure_example: 'Forgetting to define x rigorously or to prove x is in target set.'
    },
    {
      title: 'Complexity Class Distinctions',
      description: 'Differentiate P, NP, and NP-complete with examples.',
      domain: 'computer science',
      prerequisites: '[2]',
      retrieval_prompts: JSON.stringify([
        'Define P, NP, NP-hard, NP-complete.',
        'Give one reduction argument skeleton.',
        'Teach back why NP-complete does not mean impossible.'
      ]),
      canonical_example: 'SAT in NP; 3SAT NP-complete via polynomial reductions.',
      failure_example: 'Claiming NP means non-polynomial.'
    }
  ];

  const tx = db.transaction((items) => items.forEach((m) => insertMission.run(m)));
  tx(missions);
}

module.exports = db;
