import { readFileSync } from 'fs';
import type { DecisionTree, DecisionTreeNode } from './types';


const layer1: DecisionTree = JSON.parse(readFileSync('./layer1Tree_v2.json', 'utf-8'));
const layer2: DecisionTree = JSON.parse(readFileSync('./layer2Tree.json', 'utf-8'));
const layer3: DecisionTree = JSON.parse(readFileSync('./layer3Tree.json', 'utf-8'));
const layer4: DecisionTree = JSON.parse(readFileSync('./layer4Tree.json', 'utf-8'));

const unifiedNodes: Record<string, DecisionTreeNode> = {
  ...layer1.nodes,
  ...layer2.nodes,
  ...layer3.nodes,
  ...layer4.nodes,
};

const layerBoundaryMap: Record<string, string> = {
  '400': '2000',
  '2910': '3000',
  '2920': '3000',
  '2930': '3000',
  '2940': '3000',
  '2950': '3000',
  '2960': '3000',
  '3910': '4100',
  '3920': '4400',
  '3930': '4200',
  '3940': '4200',
  '3950': '4300',
};

export function getNextNodeId(
  currentNodeId: string,
  selectedOptionValue?: string
): string | null {
  const node = unifiedNodes[currentNodeId];
  if (!node) return null;

  // Screening node: red flag or default
  if (node.type === 'screening') {
    if (selectedOptionValue && selectedOptionValue !== 'none') {
      const flag = node.redFlags?.find((f) => f.id === selectedOptionValue);
      return flag?.nextNodeId || '999';
    }
    return node.defaultNextNodeId || null;
  }

  // Question node
  if (node.type === 'question' && selectedOptionValue) {
    const option = node.options?.find((o) => o.value === selectedOptionValue);
    return option?.nextNodeId || null;
  }

  // Diagnosis/Terminal node bridge
  if (node.type === 'diagnosis' && layerBoundaryMap[currentNodeId]) {
    return layerBoundaryMap[currentNodeId];
  }

  return null;
}

export function getNode(nodeId: string): DecisionTreeNode | null {
  return unifiedNodes[nodeId] || null;
}