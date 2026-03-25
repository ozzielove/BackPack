const express = require('express');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const path = require('path');
const db = require('./db');
const { toQuality, nextIntervalDays, datePlusDays, masteryState, xpFromSession } = require('./learning');

const app = express();
const PORT = process.env.PORT || 3000;
const JWT_SECRET = process.env.JWT_SECRET || 'dev-secret-change-me';

app.use(express.json());
app.use(express.static(path.join(__dirname, '..', 'public')));

function auth(req, res, next) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Missing auth token' });
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch {
    res.status(401).json({ error: 'Invalid token' });
  }
}

function levelFromXp(xp) {
  return Math.floor((xp || 0) / 300) + 1;
}

app.post('/api/auth/register', (req, res) => {
  const { email, password, name } = req.body;
  if (!email || !password || !name) return res.status(400).json({ error: 'email, password, and name are required' });

  const existing = db.prepare('SELECT id FROM users WHERE email = ?').get(email);
  if (existing) return res.status(409).json({ error: 'Email already registered' });

  const passwordHash = bcrypt.hashSync(password, 10);
  const result = db.prepare('INSERT INTO users(email, password_hash, name) VALUES(?,?,?)').run(email, passwordHash, name);
  const token = jwt.sign({ userId: result.lastInsertRowid, email }, JWT_SECRET, { expiresIn: '7d' });
  res.status(201).json({ token });
});

app.post('/api/auth/login', (req, res) => {
  const { email, password } = req.body;
  const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email);
  if (!user || !bcrypt.compareSync(password, user.password_hash)) return res.status(401).json({ error: 'Invalid credentials' });

  const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' });
  res.json({ token });
});

app.get('/api/profile', auth, (req, res) => {
  const user = db.prepare(`SELECT id, email, name, goals, domains, weekly_minutes, preferred_session_length, created_at
                           FROM users WHERE id = ?`).get(req.user.userId);
  res.json(user);
});

app.put('/api/profile', auth, (req, res) => {
  const { goals = '', domains = '', weeklyMinutes = 300, preferredSessionLength = 30 } = req.body;
  db.prepare(`UPDATE users SET goals = ?, domains = ?, weekly_minutes = ?, preferred_session_length = ? WHERE id = ?`)
    .run(goals, domains, weeklyMinutes, preferredSessionLength, req.user.userId);
  res.json({ ok: true });
});

app.get('/api/missions', auth, (req, res) => {
  const missions = db.prepare('SELECT * FROM missions').all();
  const withState = missions.map((mission) => {
    const rows = db.prepare('SELECT retrieval_score FROM sessions WHERE user_id = ? AND mission_id = ? AND status = "completed"')
      .all(req.user.userId, mission.id);
    const avg = rows.length ? rows.reduce((sum, row) => sum + row.retrieval_score, 0) / rows.length : 0;
    return {
      ...mission,
      prerequisites: JSON.parse(mission.prerequisites || '[]'),
      retrieval_prompts: JSON.parse(mission.retrieval_prompts || '[]'),
      mastery_state: masteryState(avg)
    };
  });
  res.json(withState);
});

app.get('/api/missions/:id', auth, (req, res) => {
  const mission = db.prepare('SELECT * FROM missions WHERE id = ?').get(req.params.id);
  if (!mission) return res.status(404).json({ error: 'Mission not found' });

  const threadSheet = db.prepare('SELECT content, updated_at FROM thread_sheets WHERE user_id = ? AND mission_id = ?')
    .get(req.user.userId, mission.id);

  res.json({
    ...mission,
    prerequisites: JSON.parse(mission.prerequisites || '[]'),
    retrieval_prompts: JSON.parse(mission.retrieval_prompts || '[]'),
    thread_sheet: threadSheet || { content: '', updated_at: null }
  });
});

app.post('/api/sessions/start', auth, (req, res) => {
  const { missionId, mode } = req.body;
  const validModes = ['rescue', 'standard', 'deep', 'mastery'];
  if (!missionId || !validModes.includes(mode)) return res.status(400).json({ error: 'Invalid missionId or mode' });

  const mission = db.prepare('SELECT id FROM missions WHERE id = ?').get(missionId);
  if (!mission) return res.status(404).json({ error: 'Mission not found' });

  const result = db.prepare('INSERT INTO sessions(user_id, mission_id, mode) VALUES(?,?,?)')
    .run(req.user.userId, missionId, mode);
  res.status(201).json({ sessionId: result.lastInsertRowid });
});

app.post('/api/sessions/:id/complete', auth, (req, res) => {
  const sessionId = Number(req.params.id);
  const { retrievalScore, confidence, errorLog = '', threadSheetDelta = '' } = req.body;
  const session = db.prepare('SELECT * FROM sessions WHERE id = ? AND user_id = ?').get(sessionId, req.user.userId);
  if (!session) return res.status(404).json({ error: 'Session not found' });
  if (session.status === 'completed') return res.status(409).json({ error: 'Session already completed' });

  const prior = db.prepare('SELECT retrieval_score, next_review_date FROM sessions WHERE user_id = ? AND mission_id = ? AND status = "completed" ORDER BY id DESC LIMIT 1')
    .get(req.user.userId, session.mission_id);
  const prevInterval = prior?.next_review_date
    ? Math.max(1, Math.round((new Date(prior.next_review_date) - new Date()) / (1000 * 60 * 60 * 24)))
    : 0;
  const quality = toQuality(retrievalScore, confidence);
  const intervalDays = nextIntervalDays(prevInterval, quality);
  const nextReviewDate = datePlusDays(intervalDays);

  db.prepare(`UPDATE sessions
              SET status='completed', retrieval_score=?, confidence=?, error_log=?, completed_at=datetime('now'), next_review_date=?
              WHERE id=?`)
    .run(retrievalScore, confidence, errorLog, nextReviewDate, sessionId);

  const existingSheet = db.prepare('SELECT content FROM thread_sheets WHERE user_id = ? AND mission_id = ?')
    .get(req.user.userId, session.mission_id);
  if (existingSheet) {
    db.prepare('UPDATE thread_sheets SET content = ?, updated_at = datetime(\'now\') WHERE user_id = ? AND mission_id = ?')
      .run(`${existingSheet.content}\n${threadSheetDelta}`.trim(), req.user.userId, session.mission_id);
  } else {
    db.prepare('INSERT INTO thread_sheets(user_id, mission_id, content) VALUES(?,?,?)')
      .run(req.user.userId, session.mission_id, threadSheetDelta.trim());
  }

  const lastCompleted = db.prepare('SELECT completed_at FROM sessions WHERE user_id = ? AND status = "completed" ORDER BY completed_at DESC LIMIT 1 OFFSET 1')
    .get(req.user.userId);
  const gapDays = lastCompleted?.completed_at
    ? Math.floor((Date.now() - new Date(lastCompleted.completed_at).getTime()) / (1000 * 60 * 60 * 24))
    : 0;

  const xp = xpFromSession({ retrievalScore: Number(retrievalScore) || 0, recoveredAfterGapDays: gapDays });
  db.prepare('INSERT INTO xp_events(user_id, mission_id, xp_amount, reason) VALUES(?,?,?,?)')
    .run(req.user.userId, session.mission_id, xp.total, `session_complete(q=${quality})`);

  res.json({ nextReviewDate, intervalDays, xp });
});

app.get('/api/reviews/due', auth, (req, res) => {
  const now = new Date().toISOString();
  const due = db.prepare(`
    SELECT s.id, s.mission_id, m.title, s.next_review_date
    FROM sessions s
    JOIN missions m ON m.id = s.mission_id
    WHERE s.user_id = ?
      AND s.status = 'completed'
      AND s.next_review_date IS NOT NULL
      AND s.next_review_date <= ?
    ORDER BY s.next_review_date ASC
  `).all(req.user.userId, now);
  res.json(due);
});

app.get('/api/dashboard', auth, (req, res) => {
  const totalXp = db.prepare('SELECT COALESCE(SUM(xp_amount),0) AS xp FROM xp_events WHERE user_id = ?').get(req.user.userId).xp;
  const completedSessions = db.prepare('SELECT COUNT(*) AS count FROM sessions WHERE user_id = ? AND status = "completed"').get(req.user.userId).count;
  const rescueSessions = db.prepare('SELECT COUNT(*) AS count FROM sessions WHERE user_id = ? AND mode = "rescue" AND status = "completed"').get(req.user.userId).count;
  const reentryCount = db.prepare(`
    SELECT COUNT(*) AS count
    FROM (
      SELECT id,
      julianday(completed_at) - julianday(LAG(completed_at) OVER (ORDER BY completed_at)) AS day_gap
      FROM sessions
      WHERE user_id = ? AND status = 'completed'
    ) x
    WHERE day_gap >= 2
  `).get(req.user.userId).count;

  res.json({
    totalXp,
    level: levelFromXp(totalXp),
    completedSessions,
    rescueSessions,
    recoveryReentries: reentryCount
  });
});

app.get('/api/health', (_, res) => {
  res.json({ status: 'ok', at: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`GamifyED MVP running at http://localhost:${PORT}`);
});
