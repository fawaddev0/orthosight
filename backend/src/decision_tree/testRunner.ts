import { getNode, getNextNodeId } from './engine';

function runTrace(patientName: string, answerPath: Record<string, string>) {
  console.log(`\n========================================`);
  console.log(` RUNNING TRACE: ${patientName}`);
  console.log(`========================================`);

  let currentNodeId: string | null = '100';

  while (currentNodeId) {
    const node = getNode(currentNodeId);
    if (!node) {
      console.log(`\n❌ Error: Node ${currentNodeId} missing in unified tree.`);
      break;
    }

    console.log(`\n[Node ${node.id}] (${node.type.toUpperCase()}) ${node.title}`);

    if (node.type === 'diagnosis') {
      console.log(`   Result: ${node.primaryDiagnosis}`);
      if (node.confidenceScore) console.log(`   Confidence: ${node.confidenceScore}%`);

      const nextBridgeId = getNextNodeId(currentNodeId);
      if (nextBridgeId) {
        console.log(`   🔗 Auto-bridging layer boundary -> Node ${nextBridgeId}`);
        currentNodeId = nextBridgeId;
        continue;
      } else {
        console.log(`   🏁 Terminal outcome reached.`);
        break;
      }
    }

    const selectedAnswer = answerPath[currentNodeId];

    if (node.type === 'screening') {
      if (!selectedAnswer || selectedAnswer === 'none') {
        console.log(`   Choice: No red flags selected -> default to ${node.defaultNextNodeId}`);
      } else {
        const flag = node.redFlags?.find((f) => f.id === selectedAnswer);
        console.log(`   Choice: Red flag [${selectedAnswer}] (${flag?.text})`);
      }
      currentNodeId = getNextNodeId(currentNodeId, selectedAnswer);
      continue;
    }

    if (!selectedAnswer) {
      console.log(`   ⚠️ Traversal stopped: No answer mapped for Node ${currentNodeId}`);
      break;
    }

    const option = node.options?.find((o) => o.value === selectedAnswer);
    if (!option) {
      console.log(`   ❌ Mismatch for Node ${currentNodeId}. Valid options:`, node.options?.map((o) => `"${o.value}"`));
      break;
    }

    console.log(`   Choice [${selectedAnswer}]: ${option.label}`);
    currentNodeId = getNextNodeId(currentNodeId, selectedAnswer);
  }
}

// runTrace('Patient A: Diabetic Frozen Shoulder', {
//   '100': 'none', // No red flags -> routes to 200
//   '200': '2',
//   '210': '2',
//   '220': '1',
//   '2000': '2',
//   '2200': '3',
//   '2300': '1',
//   '2410': '2',
//   '3000': '2',
//   '3100': '1',
//   '3200': '1',
//   '4000': '1',
//   '4100': '2',
//   '4110': '1'
// });

// // Patient B: Traumatic Rotator Cuff Tear
// runTrace('Patient B: Acute Traumatic Cuff Tear', {
//   '100': 'none', // No red flags -> 200
//   '200': '2',    // Negative C-spine -> 210
//   '210': '2',    // Deltoid pain -> 220
//   '220': '1',    // Night pain present -> 400
//   '2000': '1',   // Acute trauma onset -> 2100
//   '2100': '1',   // Acute <6w -> 2430
//   '2430': '1',   // High FABQ -> 2920 (Layer 2 Diagnosis)
//   '3000': '2',   // Active ROM restricted -> 3100
//   '3100': '2',   // Normal passive ROM (Active-passive discordance) -> 3300
//   '3300': '1',   // Weak/painful supraspinatus -> 3310
//   '3310': '1',   // Drop arm positive -> 3920 (Full-thickness tear)
//   '4000': '2',   // Select Node 3920 path -> 4400
//   '4400': '1',   // QuickDASH <20 -> 4910 (Ortho evaluation)
// });

// // Case 1: Red Flag Emergency (Layer 1 Exit)
// runTrace('Case 1: Red Flag Emergency (Gait Disturbance)', {
//   '100': '101' // Gait disturbance selected -> Node 999
// });

// Case 2: Cervical Spine Origin (Layer 1 Exit)
// runTrace('Case 2: Cervical Spine Radiculopathy', {
//   '100': 'none', // No red flags -> Node 200
//   '200': '1'    // Spurling / Arm Squeeze positive -> Node 300 (C-spine referral)
// });

// // Case 3: Indeterminate Physical Exam (Layer 3 Exit -> Layer 4 Trial Care)
// runTrace('Case 3: Unclustered Objective Findings (Indeterminate)', {
//   '100': 'none', // No red flags -> 200
//   '200': '2',    // Negative C-spine -> 210
//   '210': '2',    // Deltoid pain -> 220
//   '220': '1',    // Night pain present -> 400
//   '2000': '2',   // Insidious onset -> 2200
//   '2200': '3',   // Chronic >12w -> 2300
//   '2300': '2',   // No metabolic risk -> 2420
//   '2420': '2',   // Low FABQ -> 2940 (Layer 2 Diagnosis)
//   '3000': '2',   // Active ROM restricted -> 3100
//   '3100': '2',   // Passive ROM normal -> 3300
//   '3300': '2',   // Resisted strength normal -> 3400
//   '3400': '3',   // 0/3 positive cluster tests -> 3950 (Indeterminate)
//   '4000': '4',   // Select Node 3950 path -> 4300
//   '4300': '2',   // QuickDASH 20-50 -> 4830 (Conservative trial + reassess)
// });

// Case Study 1: Diabetic Adhesive Capsulitis (Butt et al., 2024)
runTrace('Real Case Study 1: Diabetic Adhesive Capsulitis (Butt et al. 2024)', {
  '100': 'none', // Normal X-ray, no red flags
  '200': '2',    // Cervical spine screen negative
  '210': '2',    // Pain localized to shoulder region
  '220': '1',    // Night pain present
  '2000': '2',   // Insidious onset
  '2200': '3',   // Chronic duration (>12 weeks)
  '2300': '1',   // Type 2 Diabetes Mellitus present
  '2410': '2',   // Low FABQ / low psychosocial risk
  '3000': '2',   // Restricted active ROM
  '3100': '1',   // Capsular pattern: passive ROM loss equals active ROM loss
  '3200': '1',   // Intact strength and neuro exam
  '4000': '1',   // Select Node 3910 path
  '4100': '2',   // Moderate disability (QuickDASH 20-50)
  '4110': '1'    // Conservative physical therapy phase
});