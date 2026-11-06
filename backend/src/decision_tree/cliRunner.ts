import * as readline from 'readline';
import { getNode, getNextNodeId } from './engine';

type DecisionTreeNode = NonNullable<ReturnType<typeof getNode>>;

const colors = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  bgBlue: '\x1b[44m',
};

interface HistoryEntry {
  nodeId: string;
  nodeTitle: string;
  selectedOptionValue?: string;
  selectedOptionLabel?: string;
}

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

function askQuestion(query: string): Promise<string> {
  return new Promise((resolve) => rl.question(query, resolve));
}

let currentNodeId: string = '100';
const historyStack: HistoryEntry[] = [];

function clearTerminal(): void {
  console.clear();
}

function getLayerFromNodeId(nodeId: string): string {
  const idNum = parseInt(nodeId, 10);
  if (idNum < 1000) return 'Layer 1: Safety & Screening';
  if (idNum < 3000) return 'Layer 2: Subjective History & Risk';
  if (idNum < 4000) return 'Layer 3: Objective Physical Exam';
  return 'Layer 4: Disability & Management';
}

function renderHeader(node: DecisionTreeNode): void {
  const layerName = getLayerFromNodeId(node.id);
  console.log(`${colors.cyan}================================================================================${colors.reset}`);
  console.log(`${colors.bold}${colors.bgBlue} OrthoSight CDSS ${colors.reset}  ${colors.bold}${layerName}${colors.reset}  [Node ID: ${colors.yellow}${node.id}${colors.reset}] (${colors.magenta}${node.type.toUpperCase()}${colors.reset})`);
  console.log(`${colors.cyan}================================================================================${colors.reset}\n`);
}

function printAuditTrail(): void {
  console.log(`\n${colors.bold}${colors.yellow}=================== CLINICAL AUDIT TRAIL / HISTORY ===================${colors.reset}`);
  if (historyStack.length === 0) {
    console.log(`${colors.dim}No steps recorded yet.${colors.reset}`);
  } else {
    historyStack.forEach((entry, index) => {
      console.log(`${colors.bold}${index + 1}. [Node ${entry.nodeId}]${colors.reset} ${entry.nodeTitle}`);
      if (entry.selectedOptionLabel) {
        console.log(`   ${colors.green}➜ Selected:${colors.reset} ${entry.selectedOptionLabel}`);
      }
    });
  }
  console.log(`${colors.yellow}======================================================================${colors.reset}\n`);
}

async function handleGlobalInput(input: string): Promise<boolean> {
  const cmd = input.trim().toUpperCase();

  if (cmd === 'B') {
    if (historyStack.length === 0) {
      console.log(`${colors.red}\n⚠️ Already at root node. Cannot go back.${colors.reset}`);
      await askQuestion('Press Enter to continue...');
    } else {
      const prev = historyStack.pop();
      if (prev) {
        currentNodeId = prev.nodeId;
      }
    }
    return true;
  }

  if (cmd === 'H') {
    printAuditTrail();
    await askQuestion('Press Enter to return to active evaluation...');
    return true;
  }

  if (cmd === 'R') {
    historyStack.length = 0;
    currentNodeId = '100';
    console.log(`${colors.green}\n🔄 Case evaluation restarted.${colors.reset}`);
    await askQuestion('Press Enter to begin...');
    return true;
  }

  if (cmd === 'Q') {
    console.log(`${colors.bold}\nExiting OrthoSight CDSS. Goodbye!${colors.reset}\n`);
    rl.close();
    process.exit(0);
  }

  return false;
}

function renderGlobalOptions(): void {
  console.log(`\n${colors.dim}────────────────────────────────────────────────────────────────────────────────${colors.reset}`);
  console.log(`${colors.bold}GLOBAL COMMANDS:${colors.reset} [B] Go Back  |  [H] View History  |  [R] Restart Case  |  [Q] Quit`);
  console.log(`${colors.dim}────────────────────────────────────────────────────────────────────────────────${colors.reset}`);
}

async function runCli(): Promise<void> {
  while (true) {
    clearTerminal();
    const node = getNode(currentNodeId);

    if (!node) {
      console.log(`${colors.red}❌ Error: Node ID "${currentNodeId}" not found in unified tree schema.${colors.reset}`);
      const handled = await askQuestion('Press [R] to restart or [Q] to quit: ');
      await handleGlobalInput(handled);
      continue;
    }

    renderHeader(node);
    console.log(`${colors.bold}${colors.green}${node.title}${colors.reset}\n`);

    // 1. SCREENING NODE
    if (node.type === 'screening') {
      console.log(`Review red flags below:`);
      console.log(`  ${colors.bold}[1]${colors.reset} ${colors.green}No Red Flags Present${colors.reset} (Proceed to default path)`);

      const redFlags = node.redFlags || [];
      redFlags.forEach((flag, idx) => {
        console.log(`  ${colors.bold}[${idx + 2}]${colors.reset} ${colors.red}⚠️  ${flag.text}${colors.reset}`);
      });

      renderGlobalOptions();

      const answer = await askQuestion(`Select option (1-${redFlags.length + 1}) or command: `);
      if (await handleGlobalInput(answer)) continue;

      const choiceNum = parseInt(answer.trim(), 10);
      if (isNaN(choiceNum) || choiceNum < 1 || choiceNum > redFlags.length + 1) {
        console.log(`${colors.red}Invalid selection.${colors.reset}`);
        await askQuestion('Press Enter to retry...');
        continue;
      }

      if (choiceNum === 1) {
        historyStack.push({
          nodeId: currentNodeId,
          nodeTitle: node.title,
          selectedOptionValue: 'none',
          selectedOptionLabel: 'No Red Flags Present',
        });
        const nextId = getNextNodeId(currentNodeId, 'none');
        if (nextId) currentNodeId = nextId;
      } else {
        const selectedFlag = redFlags[choiceNum - 2];
        if (!selectedFlag) continue;

        historyStack.push({
          nodeId: currentNodeId,
          nodeTitle: node.title,
          selectedOptionValue: selectedFlag.id,
          selectedOptionLabel: `Red Flag: ${selectedFlag.text}`,
        });
        const nextId = getNextNodeId(currentNodeId, selectedFlag.id);
        if (nextId) currentNodeId = nextId;
      }
      continue;
    }

    // 2. QUESTION NODE
    if (node.type === 'question') {
      if (node.questionText) {
        console.log(`${colors.dim}${node.questionText}${colors.reset}\n`);
      }

      const options = node.options || [];
      options.forEach((opt, idx) => {
        console.log(`  ${colors.bold}[${idx + 1}]${colors.reset} ${opt.label}`);
      });

      renderGlobalOptions();

      const answer = await askQuestion(`Select choice (1-${options.length}) or command: `);
      if (await handleGlobalInput(answer)) continue;

      const choiceNum = parseInt(answer.trim(), 10);
      if (isNaN(choiceNum) || choiceNum < 1 || choiceNum > options.length) {
        console.log(`${colors.red}Invalid choice number.${colors.reset}`);
        await askQuestion('Press Enter to retry...');
        continue;
      }

      const selectedOpt = options[choiceNum - 1];
      if (!selectedOpt) continue;

      historyStack.push({
        nodeId: currentNodeId,
        nodeTitle: node.title,
        selectedOptionValue: selectedOpt.value,
        selectedOptionLabel: selectedOpt.label,
      });

      const nextId = getNextNodeId(currentNodeId, selectedOpt.value);
      if (nextId) currentNodeId = nextId;
      continue;
    }

    // 3. DIAGNOSIS NODE
    if (node.type === 'diagnosis') {
      console.log(`┌──────────────────────────────────────────────────────────────────────────────┐`);
      console.log(`│ ${colors.bold}${colors.magenta}PRIMARY EVALUATION OUTCOME${colors.reset}                                           │`);
      console.log(`├──────────────────────────────────────────────────────────────────────────────┤`);
      console.log(`│ ${colors.bold}${node.primaryDiagnosis}${colors.reset}`);
      if (node.confidenceScore !== undefined) {
        console.log(`│ ${colors.cyan}Diagnostic Confidence Score: ${node.confidenceScore}%${colors.reset}`);
      }
      if (node.differentials && node.differentials.length > 0) {
        console.log(`│`);
        console.log(`│ ${colors.yellow}Differential Diagnoses:${colors.reset}`);
        node.differentials.forEach((diff) => {
          console.log(`│   • ${diff}`);
        });
      }
      console.log(`└──────────────────────────────────────────────────────────────────────────────┘\n`);

      const nextBridgeId = getNextNodeId(currentNodeId);

      if (nextBridgeId) {
        console.log(`${colors.yellow}🔗 Transitioning to next assessment layer (Node ${nextBridgeId})...${colors.reset}`);
        renderGlobalOptions();
        const input = await askQuestion(`Press [Enter] to proceed, or enter global command: `);
        if (await handleGlobalInput(input)) continue;

        historyStack.push({
          nodeId: currentNodeId,
          nodeTitle: node.title,
          selectedOptionLabel: `Auto-bridged outcome -> Node ${nextBridgeId}`,
        });
        currentNodeId = nextBridgeId;
        continue;
      }

      console.log(`${colors.bold}${colors.green}🏁 TERMINAL EVALUATION OUTCOME REACHED${colors.reset}\n`);
      renderGlobalOptions();
      const choice = await askQuestion(`Select option [H/R/Q]: `);
      if (await handleGlobalInput(choice)) continue;
    }
  }
}

runCli().catch((err) => {
  console.error(`${colors.red}Fatal Error:${colors.reset}`, err);
  process.exit(1);
});