import type { Condition, RuleGroup } from './ruleTypes';

export function evaluateCondition(
  patientData: Record<string, unknown>,
  condition: Condition,
): boolean {
  if (!Object.prototype.hasOwnProperty.call(patientData, condition.field)) {
    return false;
  }

  const actual = patientData[condition.field];
  const expected = condition.value;

  switch (condition.operator) {
    case 'EQUALS':
      return actual === expected;
    case 'NOT_EQUALS':
      return actual !== expected;
    case 'GREATER_THAN':
      return typeof actual === 'number' && typeof expected === 'number'
        ? actual > expected
        : false;
    case 'LESS_THAN':
      return typeof actual === 'number' && typeof expected === 'number'
        ? actual < expected
        : false;
    case 'CONTAINS':
      if (typeof actual === 'string' && typeof expected === 'string') {
        return actual.includes(expected);
      }
      return Array.isArray(actual) && actual.some((item: unknown) => item === expected);
  }
}

export function evaluateRuleGroup(
  patientData: Record<string, unknown>,
  group: RuleGroup,
): boolean {
  const evaluate = (item: Condition | RuleGroup): boolean =>
    'logic' in item
      ? evaluateRuleGroup(patientData, item)
      : evaluateCondition(patientData, item);

  return group.logic === 'AND'
    ? group.conditions.every(evaluate)
    : group.conditions.some(evaluate);
}
