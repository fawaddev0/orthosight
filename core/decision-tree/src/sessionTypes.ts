export interface HistoryEntry {
  nodeId: string;
  nodeType: string;
  selectedAnswer: string;
  timestamp: number;
}

export interface AssessmentSession {
  sessionId: string;
  treeId: string;
  currentNodeId: string;
  history: HistoryEntry[];
  answers: Record<string, string>;
  isComplete: boolean;
  finalDiagnosisId?: string;
}
