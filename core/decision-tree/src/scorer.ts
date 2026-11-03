import { evaluateRuleGroup } from './ruleEvaluator';
import type { DiagnosticScoreRule } from './ruleTypes';

export interface ScoreResult {
  diagnosisId: string;
  score: number;
  confidence: number;
  matchedCriteriaCount: number;
}

export function calculateDiagnosticScores(
  patientData: Record<string, unknown>,
  rules: DiagnosticScoreRule[],
): ScoreResult[] {
  const totals = new Map<
    string,
    { score: number; possibleWeight: number; matchedCriteriaCount: number }
  >();

  for (const rule of rules) {
    const total = totals.get(rule.diagnosisId) ?? {
      score: 0,
      possibleWeight: 0,
      matchedCriteriaCount: 0,
    };

    total.possibleWeight += Math.max(0, rule.weight);
    if (evaluateRuleGroup(patientData, rule.criteria)) {
      total.score += rule.weight;
      total.matchedCriteriaCount += 1;
    }
    totals.set(rule.diagnosisId, total);
  }

  return Array.from(totals, ([diagnosisId, total]): ScoreResult => ({
    diagnosisId,
    score: total.score,
    confidence:
      total.possibleWeight > 0
        ? Math.max(0, Math.min(100, (total.score / total.possibleWeight) * 100))
        : 0,
    matchedCriteriaCount: total.matchedCriteriaCount,
  })).sort((left, right) =>
    right.score - left.score || left.diagnosisId.localeCompare(right.diagnosisId),
  );
}
