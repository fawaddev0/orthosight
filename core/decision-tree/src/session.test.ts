import { describe, expect, it } from 'vitest';
import { mockTree } from './mockTree';
import {
  answerCurrentNode,
  createSession,
  EmptyHistoryError,
  getSessionSummary,
  SessionAlreadyCompletedError,
  undoLastAnswer,
} from './session';

describe('assessment session', () => {
  it('starts at the root with empty history and answers', () => {
    const session = createSession(mockTree, 'session-1');

    expect(session).toEqual({
      sessionId: 'session-1',
      treeId: mockTree.id,
      currentNodeId: 'root',
      history: [],
      answers: {},
      isComplete: false,
    });
  });

  it('records an answer and advances without mutating the previous state', () => {
    const initial = createSession(mockTree, 'session-1');
    const next = answerCurrentNode(mockTree, initial, 'no');

    expect(next.currentNodeId).toBe('subjective_q1');
    expect(next.history).toEqual([
      {
        nodeId: 'root',
        nodeType: 'screening',
        selectedAnswer: 'no',
        timestamp: expect.any(Number),
      },
    ]);
    expect(next.answers).toEqual({ root: 'no' });
    expect(initial.history).toEqual([]);
    expect(initial.answers).toEqual({});
  });

  it('marks a session complete on reaching a diagnosis', () => {
    const session = answerCurrentNode(mockTree, createSession(mockTree), 'trauma');

    expect(session.isComplete).toBe(true);
    expect(session.finalDiagnosisId).toBe('emergency_node');
  });

  it('rejects answers after completion', () => {
    const session = answerCurrentNode(mockTree, createSession(mockTree), 'trauma');

    expect(() => answerCurrentNode(mockTree, session, 'anything')).toThrow(
      SessionAlreadyCompletedError,
    );
  });

  it('undoes the last answer and reopens a completed session', () => {
    const initial = createSession(mockTree);
    const completed = answerCurrentNode(mockTree, initial, 'trauma');
    const undone = undoLastAnswer(mockTree, completed);

    expect(undone.currentNodeId).toBe('root');
    expect(undone.history).toEqual([]);
    expect(undone.answers).toEqual({});
    expect(undone.isComplete).toBe(false);
    expect(undone.finalDiagnosisId).toBeUndefined();
    expect(completed.history).toHaveLength(1);
    expect(completed.answers).toEqual({ root: 'trauma' });
  });

  it('keeps earlier answers when undoing the latest step', () => {
    const first = answerCurrentNode(mockTree, createSession(mockTree), 'no');
    const second = answerCurrentNode(mockTree, first, 'recent');
    const undone = undoLastAnswer(mockTree, second);

    expect(undone.currentNodeId).toBe('subjective_q1');
    expect(undone.history).toEqual(first.history);
    expect(undone.answers).toEqual({ root: 'no' });
    expect(undone.isComplete).toBe(false);
    expect(undone.finalDiagnosisId).toBeUndefined();
  });

  it('rejects undo on a session with no history', () => {
    expect(() => undoLastAnswer(mockTree, createSession(mockTree))).toThrow(
      EmptyHistoryError,
    );
  });

  it('summarizes the path, answers, and final diagnosis', () => {
    const session = answerCurrentNode(mockTree, createSession(mockTree), 'infection');

    expect(getSessionSummary(mockTree, session)).toEqual({
      path: ['root', 'emergency_node'],
      answers: { root: 'infection' },
      finalDiagnosis: mockTree.nodes.emergency_node,
    });
  });
});
