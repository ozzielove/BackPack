const test = require('node:test');
const assert = require('node:assert/strict');
const { toQuality, nextIntervalDays, masteryState, xpFromSession } = require('../src/learning');

test('toQuality converts retrieval+confidence into 0-5 quality', () => {
  assert.equal(toQuality(1, 1), 5);
  assert.equal(toQuality(0, 0), 0);
});

test('nextIntervalDays grows interval for quality >= 3', () => {
  assert.equal(nextIntervalDays(0, 4), 2);
  assert.equal(nextIntervalDays(2, 4), 4);
  assert.ok(nextIntervalDays(4, 5) > 4);
});

test('masteryState maps average score bands', () => {
  assert.equal(masteryState(0.5), 'Fragile');
  assert.equal(masteryState(0.75), 'Transfer-ready');
  assert.equal(masteryState(0.95), 'Teach-ready');
});

test('xpFromSession includes recovery and repair bonuses', () => {
  const xp = xpFromSession({ retrievalScore: 0.6, recoveredAfterGapDays: 3 });
  assert.equal(xp.total, 110);
});
