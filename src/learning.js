function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function toQuality(retrievalScore, confidence) {
  const score = clamp(Number(retrievalScore) || 0, 0, 1);
  const conf = clamp(Number(confidence) || 0, 0, 1);
  return Math.round((score * 0.75 + conf * 0.25) * 5);
}

function nextIntervalDays(previousInterval, quality) {
  if (quality < 3) return 1;
  if (!previousInterval || previousInterval <= 1) return 2;
  if (previousInterval <= 2) return 4;
  const easiness = clamp(1.3 + quality * 0.15, 1.3, 2.8);
  return Math.round(previousInterval * easiness);
}

function datePlusDays(days) {
  const date = new Date();
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString();
}

function masteryState(avgScore) {
  if (avgScore >= 0.9) return 'Teach-ready';
  if (avgScore >= 0.8) return 'Operational';
  if (avgScore >= 0.7) return 'Transfer-ready';
  if (avgScore >= 0.55) return 'Stable';
  return 'Fragile';
}

function xpFromSession({ retrievalScore, recoveredAfterGapDays }) {
  const base = Math.round(clamp(retrievalScore, 0, 1) * 100);
  const repairBonus = retrievalScore < 0.7 ? 20 : 0;
  const recoveryBonus = recoveredAfterGapDays >= 2 ? 30 : 0;
  return {
    total: base + repairBonus + recoveryBonus,
    breakdown: { base, repairBonus, recoveryBonus }
  };
}

module.exports = {
  toQuality,
  nextIntervalDays,
  datePlusDays,
  masteryState,
  xpFromSession
};
