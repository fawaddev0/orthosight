import { describe, expect, it } from 'vitest';
import { evaluateCondition, evaluateRuleGroup } from './ruleEvaluator';
import type { DiagnosticScoreRule, RuleGroup } from './ruleTypes';
import { calculateDiagnosticScores } from './scorer';

describe('rule evaluation', () => {
  it('compares numeric values', () => {
    expect(
      evaluateCondition({ age: 52 }, { field: 'age', operator: 'GREATER_THAN', value: 50 }),
    ).toBe(true);
    expect(
      evaluateCondition({ age: 52 }, { field: 'age', operator: 'LESS_THAN', value: 50 }),
    ).toBe(false);
  });

  it('matches strings with EQUALS and CONTAINS', () => {
    expect(
      evaluateCondition(
        { symptom: 'shoulder pain' },
        { field: 'symptom', operator: 'EQUALS', value: 'shoulder pain' },
      ),
    ).toBe(true);
    expect(
      evaluateCondition(
        { symptom: 'shoulder pain' },
        { field: 'symptom', operator: 'CONTAINS', value: 'shoulder' },
      ),
    ).toBe(true);
  });

  it('compares boolean flags', () => {
    expect(
      evaluateCondition(
        { fever: false },
        { field: 'fever', operator: 'NOT_EQUALS', value: true },
      ),
    ).toBe(true);
    expect(
      evaluateCondition(
        { fever: false },
        { field: 'fever', operator: 'EQUALS', value: false },
      ),
    ).toBe(true);
  });

  it('evaluates nested AND and OR groups', () => {
    const criteria: RuleGroup = {
      logic: 'AND',
      conditions: [
        { field: 'pain', operator: 'EQUALS', value: true },
        {
          logic: 'OR',
          conditions: [
            { field: 'age', operator: 'GREATER_THAN', value: 50 },
            { field: 'trauma', operator: 'EQUALS', value: true },
          ],
        },
      ],
    };

    expect(evaluateRuleGroup({ pain: true, age: 30, trauma: true }, criteria)).toBe(true);
    expect(evaluateRuleGroup({ pain: true, age: 30, trauma: false }, criteria)).toBe(false);
  });
});

describe('diagnostic scoring', () => {
  const rules: DiagnosticScoreRule[] = [
    {
      diagnosisId: 'rotator_cuff',
      weight: 4,
      criteria: {
        logic: 'AND',
        conditions: [{ field: 'pain', operator: 'EQUALS', value: true }],
      },
    },
    {
      diagnosisId: 'rotator_cuff',
      weight: 2,
      criteria: {
        logic: 'AND',
        conditions: [{ field: 'weakness', operator: 'EQUALS', value: true }],
      },
    },
    {
      diagnosisId: 'capsulitis',
      weight: 3,
      criteria: {
        logic: 'AND',
        conditions: [{ field: 'stiffness', operator: 'EQUALS', value: true }],
      },
    },
  ];

  it('ranks diagnoses by matched weight and normalizes confidence', () => {
    const scores = calculateDiagnosticScores(
      { pain: true, weakness: false, stiffness: true },
      rules,
    );

    expect(scores.map((result) => result.diagnosisId)).toEqual([
      'rotator_cuff',
      'capsulitis',
    ]);
    expect(scores[0]).toMatchObject({ score: 4, matchedCriteriaCount: 1 });
    expect(scores[0]?.confidence).toBeCloseTo(66.67, 2);
    expect(scores[1]).toEqual({
      diagnosisId: 'capsulitis',
      score: 3,
      confidence: 100,
      matchedCriteriaCount: 1,
    });
  });

  it('returns zero scores for diagnoses with no matching rules', () => {
    expect(calculateDiagnosticScores({}, rules)).toEqual([
      {
        diagnosisId: 'capsulitis',
        score: 0,
        confidence: 0,
        matchedCriteriaCount: 0,
      },
      {
        diagnosisId: 'rotator_cuff',
        score: 0,
        confidence: 0,
        matchedCriteriaCount: 0,
      },
    ]);
  });
});
