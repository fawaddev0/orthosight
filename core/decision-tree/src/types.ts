export interface BaseNode {
  id: string;
  type: 'screening' | 'question' | 'diagnosis';
  title: string;
}

export interface ScreeningNode extends BaseNode {
  type: 'screening';
  redFlags: Array<{ id: string; text: string; nextNodeId: string }>;
  defaultNextNodeId: string;
}

export interface QuestionNode extends BaseNode {
  type: 'question';
  questionText: string;
  options: Array<{ value: string; label: string; nextNodeId: string }>;
}

export interface DiagnosisNode extends BaseNode {
  type: 'diagnosis';
  primaryDiagnosis: string;
  differentials: string[];
  confidenceScore: number;
  icdCode?: string;
}

export type DecisionTreeNode = ScreeningNode | QuestionNode | DiagnosisNode;

export interface DecisionTree {
  id: string;
  version: string;
  rootNodeId: string;
  nodes: Record<string, DecisionTreeNode>;
}
