import { describe, expect, it } from 'vitest';
import {
  getNextNode,
  InvalidAnswerError,
  NodeNotFoundError,
  TerminalNodeError,
} from './engine';
import { mockTree } from './mockTree';
import { validateDecisionTree } from './schema';

describe('decision tree', () => {
  it('validates the mock tree', () => {
    expect(validateDecisionTree(mockTree)).toEqual(mockTree);
  });

  it('routes a trauma red flag to the emergency node', () => {
    expect(getNextNode(mockTree, 'root', 'trauma').id).toBe('emergency_node');
  });

  it('routes a cleared red flag screen to the onset question', () => {
    expect(getNextNode(mockTree, 'root', 'no').id).toBe('subjective_q1');
  });

  it('throws NodeNotFoundError for an unknown current node', () => {
    expect(() => getNextNode(mockTree, 'unknown', 'no')).toThrow(NodeNotFoundError);
  });

  it('throws InvalidAnswerError for an unknown question option', () => {
    expect(() => getNextNode(mockTree, 'subjective_q1', 'unknown')).toThrow(
      InvalidAnswerError,
    );
  });

  it('throws TerminalNodeError when traversing past a diagnosis', () => {
    expect(() => getNextNode(mockTree, 'emergency_node', 'anything')).toThrow(
      TerminalNodeError,
    );
  });
});
