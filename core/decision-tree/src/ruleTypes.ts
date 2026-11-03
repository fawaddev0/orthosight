export type Operator =
  | 'EQUALS'
  | 'NOT_EQUALS'
  | 'GREATER_THAN'
  | 'LESS_THAN'
  | 'CONTAINS';

export interface Condition {
  field: string;
  operator: Operator;
  value: string | number | boolean;
}

export interface RuleGroup {
  logic: 'AND' | 'OR';
  conditions: Array<Condition | RuleGroup>;
}

export interface DiagnosticScoreRule {
  diagnosisId: string;
  weight: number;
  criteria: RuleGroup;
}
