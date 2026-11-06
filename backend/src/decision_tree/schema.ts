import { z } from 'zod';
import type { DecisionTree } from './types';

export const BaseNodeSchema = z.object({
  id: z.string(),
  type: z.enum(['screening', 'question', 'diagnosis']),
  title: z.string(),
});

export const ScreeningNodeSchema = z.object({
  id: z.string(),
  type: z.literal('screening'),
  title: z.string(),
  redFlags: z.array(
    z.object({
      id: z.string(),
      text: z.string(),
      nextNodeId: z.string(),
    }),
  ),
  defaultNextNodeId: z.string(),
});

export const QuestionNodeSchema = z.object({
  id: z.string(),
  type: z.literal('question'),
  title: z.string(),
  questionText: z.string(),
  options: z.array(
    z.object({
      value: z.string(),
      label: z.string(),
      nextNodeId: z.string(),
    }),
  ),
});

export const DiagnosisNodeSchema = z.object({
  id: z.string(),
  type: z.literal('diagnosis'),
  title: z.string(),
  primaryDiagnosis: z.string(),
  differentials: z.array(z.string()),
  confidenceScore: z.number(),
  icdCode: z.string().optional(),
});

export const DecisionTreeNodeSchema = z.discriminatedUnion('type', [
  ScreeningNodeSchema,
  QuestionNodeSchema,
  DiagnosisNodeSchema,
]);

export const DecisionTreeSchema = z.object({
  id: z.string(),
  version: z.string(),
  rootNodeId: z.string(),
  nodes: z.record(DecisionTreeNodeSchema),
});

export function validateDecisionTree(data: unknown): DecisionTree {
  return DecisionTreeSchema.parse(data);
}
