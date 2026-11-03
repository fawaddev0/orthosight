import { getNextNode, getNode } from './engine';
import type { AssessmentSession, HistoryEntry } from './sessionTypes';
import type { DecisionTree, DiagnosisNode } from './types';

export class SessionAlreadyCompletedError extends Error {
  constructor(sessionId: string) {
    super(`Session already completed: ${sessionId}`);
    this.name = 'SessionAlreadyCompletedError';
  }
}

export class EmptyHistoryError extends Error {
  constructor(sessionId: string) {
    super(`Session has no answers to undo: ${sessionId}`);
    this.name = 'EmptyHistoryError';
  }
}

export function createSession(
  tree: DecisionTree,
  sessionId: string = globalThis.crypto.randomUUID(),
): AssessmentSession {
  return {
    sessionId,
    treeId: tree.id,
    currentNodeId: tree.rootNodeId,
    history: [],
    answers: {},
    isComplete: false,
  };
}

export function answerCurrentNode(
  tree: DecisionTree,
  session: AssessmentSession,
  answer: string,
): AssessmentSession {
  if (session.isComplete) {
    throw new SessionAlreadyCompletedError(session.sessionId);
  }

  const currentNode = getNode(tree, session.currentNodeId);
  const nextNode = getNextNode(tree, session.currentNodeId, answer);
  const entry: HistoryEntry = {
    nodeId: currentNode.id,
    nodeType: currentNode.type,
    selectedAnswer: answer,
    timestamp: Date.now(),
  };
  const isComplete = nextNode.type === 'diagnosis';

  return {
    ...session,
    currentNodeId: nextNode.id,
    history: [...session.history, entry],
    answers: { ...session.answers, [currentNode.id]: answer },
    isComplete,
    finalDiagnosisId: isComplete ? nextNode.id : undefined,
  };
}

export function undoLastAnswer(
  tree: DecisionTree,
  session: AssessmentSession,
): AssessmentSession {
  const lastEntry = session.history.at(-1);
  if (!lastEntry) {
    throw new EmptyHistoryError(session.sessionId);
  }

  getNode(tree, lastEntry.nodeId);
  const answers = { ...session.answers };
  delete answers[lastEntry.nodeId];

  return {
    ...session,
    currentNodeId: lastEntry.nodeId,
    history: session.history.slice(0, -1),
    answers,
    isComplete: false,
    finalDiagnosisId: undefined,
  };
}

export interface SessionSummary {
  path: string[];
  answers: Record<string, string>;
  finalDiagnosis?: DiagnosisNode;
}

export function getSessionSummary(
  tree: DecisionTree,
  session: AssessmentSession,
): SessionSummary {
  const path = [...session.history.map((entry) => entry.nodeId), session.currentNodeId];
  const answers = { ...session.answers };

  if (!session.isComplete) {
    return { path, answers };
  }

  const finalNode = getNode(tree, session.finalDiagnosisId ?? session.currentNodeId);
  if (finalNode.type !== 'diagnosis') {
    throw new Error(`Final node is not a diagnosis: ${finalNode.id}`);
  }

  return { path, answers, finalDiagnosis: finalNode };
}
