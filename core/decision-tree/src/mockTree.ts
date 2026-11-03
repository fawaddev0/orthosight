import type { DecisionTree } from './types';

export const mockTree: DecisionTree = {
  id: 'shoulder_pain_v1',
  version: '1.0.0',
  rootNodeId: 'root',
  nodes: {
    root: {
      id: 'root',
      type: 'screening',
      title: 'Red flag check',
      redFlags: [
        { id: 'trauma', text: 'Recent trauma', nextNodeId: 'emergency_node' },
        { id: 'infection', text: 'Signs of infection', nextNodeId: 'emergency_node' },
      ],
      defaultNextNodeId: 'subjective_q1',
    },
    emergency_node: {
      id: 'emergency_node',
      type: 'diagnosis',
      title: 'Red flag referral',
      primaryDiagnosis: 'Possible urgent shoulder pathology',
      differentials: ['Traumatic injury', 'Infection'],
      confidenceScore: 0,
    },
    subjective_q1: {
      id: 'subjective_q1',
      type: 'question',
      title: 'Onset duration',
      questionText: 'How long have you had shoulder pain?',
      options: [
        { value: 'recent', label: 'Less than two weeks', nextNodeId: 'emergency_node' },
        { value: 'persistent', label: 'Two weeks or longer', nextNodeId: 'emergency_node' },
      ],
    },
  },
};
