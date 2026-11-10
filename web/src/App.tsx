import { useEffect, useState } from 'react'
import { Activity, ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react'
import { Button } from '../components/ui/button'
import { QuickDashQuestionnaire } from './components/QuickDashQuestionnaire'
import './App.css'

interface RedFlag {
  id: string
  text: string
  nextNodeId: string
}

interface Option {
  value: string
  label: string
  nextNodeId: string
}

interface ScreeningNode {
  id: string
  type: 'screening'
  title: string
  redFlags: RedFlag[]
  defaultNextNodeId: string
}

interface QuestionNode {
  id: string
  type: 'question'
  title: string
  questionText: string
  options: Option[]
}

interface QuestionnaireEvalNode {
  id: string
  type: 'questionnaire_eval'
  title: string
  questionText: string
  options: Option[]
}

interface DiagnosisNode {
  id: string
  type: 'diagnosis'
  title: string
  primaryDiagnosis: string
  differentials: string[]
  confidenceScore: number
  icdCode?: string | null
}

type DecisionNode = ScreeningNode | QuestionNode | QuestionnaireEvalNode | DiagnosisNode

interface QuestionnaireItem {
  id: number
  text: string
}

const FABQ_ITEMS: QuestionnaireItem[] = [
  { id: 1, text: 'My pain was caused by physical activity.' },
  { id: 2, text: 'Physical activity makes my pain worse.' },
  { id: 3, text: 'Physical activity might harm my shoulder.' },
  { id: 4, text: 'I should not do physical activities which might make my pain worse.' },
  { id: 5, text: 'I cannot do physical activities which might make my pain worse.' },
  { id: 6, text: 'My pain was caused by my work or by an accident at work.' },
  { id: 7, text: 'My work makes my pain worse.' },
  { id: 8, text: 'I have a claim for compensation for my pain.' },
  { id: 9, text: 'My work is too heavy for me.' },
  { id: 10, text: 'My work makes or would make my pain worse.' },
  { id: 11, text: 'My work might harm my shoulder.' },
  { id: 12, text: 'I should not do my normal work with my present pain.' },
  { id: 13, text: 'I cannot do my normal work with my present pain.' },
  { id: 14, text: 'I cannot do my normal work until my pain is treated.' },
  { id: 15, text: 'I do not think that I will be able to go back to my normal work within 3 months.' },
  { id: 16, text: 'I do not think that I will ever be able to go back to my work.' },
]

const QUICKDASH_ITEMS: QuestionnaireItem[] = [
  { id: 1, text: 'Open a tight or new jar.' },
  { id: 2, text: 'Do heavy household chores (e.g., wash walls, floors).' },
  { id: 3, text: 'Carry a shopping bag or briefcase.' },
  { id: 4, text: 'Wash your back.' },
  { id: 5, text: 'Use a knife to cut food.' },
  { id: 6, text: 'Recreational activities in which you take some force or impact through your arm, shoulder or hand.' },
  { id: 7, text: 'Extent arm, shoulder or hand problem interfered with normal social activities.' },
  { id: 8, text: 'Limitation in work or other regular daily activities.' },
  { id: 9, text: 'Arm, shoulder or hand pain severity.' },
  { id: 10, text: 'Tingling (pins and needles) in arm, shoulder or hand.' },
  { id: 11, text: 'Difficulty sleeping because of pain in arm, shoulder or hand.' },
]

interface DecisionTreeResponse {
  node: DecisionNode
  nextNode: DecisionNode | null
  route: {
    sourceNodeId: string
    answer: string | null
    targetNodeId: string
  } | null
  diagnosis: DiagnosisNode | null
  complete: boolean
}

interface DecisionTreeRequest {
  nodeId?: string
  answer?: string
  fabqAnswers?: Record<number, number>
  quickdashAnswers?: Record<number, number>
}

const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

async function requestDecisionTree(
  payload: DecisionTreeRequest = {},
): Promise<DecisionTreeResponse> {
  const response = await fetch(`${apiBaseUrl}/api/decision-tree`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const errorBody: unknown = await response.json().catch(() => null)
    const detail =
      typeof errorBody === 'object' && errorBody !== null && 'detail' in errorBody
        ? errorBody.detail
        : null
    throw new Error(
      typeof detail === 'string'
        ? detail
        : 'We could not load this step. Check your connection and try again.',
    )
  }

  return (await response.json()) as DecisionTreeResponse
}

function layerForNode(node: DecisionNode): number {
  const id = Number(node.id)
  return id >= 1000 ? Math.floor(id / 1000) : 1
}

function App() {
  const [node, setNode] = useState<DecisionNode | null>(null)
  const [diagnosis, setDiagnosis] = useState<DiagnosisNode | null>(null)
  const [history, setHistory] = useState<DecisionNode[]>([])
  const [selectedRedFlag, setSelectedRedFlag] = useState<string | null>(null)
  const [formAnswers, setFormAnswers] = useState<Record<number, number>>({})
  const [isLoading, setIsLoading] = useState(true)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let isCurrent = true

    requestDecisionTree()
      .then((result) => {
        if (isCurrent) setNode(result.node)
      })
      .catch((requestError: unknown) => {
        if (isCurrent) {
          setError(
            requestError instanceof Error
              ? requestError.message
              : 'We could not start the assessment. Try again.',
          )
        }
      })
      .finally(() => {
        if (isCurrent) setIsLoading(false)
      })

    return () => {
      isCurrent = false
    }
  }, [])

  async function followAnswer(
    answer?: string,
    fabqAnswers?: Record<number, number>,
    quickdashAnswers?: Record<number, number>,
  ) {
    if (node === null) return

    setIsSubmitting(true)
    setError(null)

    try {
      let result = await requestDecisionTree({
        nodeId: node.id,
        answer,
        fabqAnswers,
        quickdashAnswers,
      })
      let nextNode = result.nextNode
      let finalDiagnosis = result.diagnosis
      let bridgeCount = 0

      while (finalDiagnosis === null && nextNode?.type === 'diagnosis' && bridgeCount < 4) {
        result = await requestDecisionTree({ nodeId: nextNode.id })
        nextNode = result.nextNode
        finalDiagnosis = result.diagnosis
        bridgeCount += 1
      }

      if (finalDiagnosis !== null) {
        setDiagnosis(finalDiagnosis)
        setHistory((steps) => [...steps, node])
        setSelectedRedFlag(null)
        setFormAnswers({})
        return
      }

      if (nextNode === null) {
        throw new Error('This path has no next step. Restart the assessment.')
      }

      setHistory((steps) => [...steps, node])
      setNode(nextNode)
      setSelectedRedFlag(null)
      setFormAnswers({})
    } catch (requestError: unknown) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'We could not save your answer. Try again.',
      )
    } finally {
      setIsSubmitting(false)
    }
  }

  async function restart() {
    setIsLoading(true)
    setDiagnosis(null)
    setNode(null)
    setHistory([])
    setSelectedRedFlag(null)
    setFormAnswers({})
    setError(null)

    try {
      const result = await requestDecisionTree()
      setNode(result.node)
    } catch (requestError: unknown) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : 'We could not start the assessment. Try again.',
      )
    } finally {
      setIsLoading(false)
    }
  }

  function goBack() {
    const previousNode = history.at(-1)
    if (previousNode === undefined) return

    setHistory((steps) => steps.slice(0, -1))
    setNode(previousNode)
    setDiagnosis(null)
    setSelectedRedFlag(null)
    setFormAnswers({})
    setError(null)
  }

  const layer = node === null ? 1 : layerForNode(node)

  return (
    <main className="assessment-shell">
      <header className="site-header">
        <a className="brand" href="/" aria-label="OrthoSight home">
          <span className="brand-mark" aria-hidden="true">
            <Activity size={19} strokeWidth={2.2} />
          </span>
          <span>OrthoSight</span>
        </a>
      </header>

      <section className="assessment-content" aria-live="polite">
        {isLoading ? (
          <div className="state-message">
            <span className="loading-mark" aria-hidden="true" />
            <p>Loading assessment</p>
          </div>
        ) : diagnosis !== null ? (
          <div className="result-view">
            <div className="result-icon" aria-hidden="true">
              <Activity size={25} strokeWidth={1.8} />
            </div>
            <p className="eyebrow">Assessment complete</p>
            <h1>{diagnosis.title}</h1>
            <p className="result-copy">{diagnosis.primaryDiagnosis}</p>
            <Button className="restart-button" onClick={restart} disabled={isSubmitting}>
              <RotateCcw size={16} />
              Start again
            </Button>
          </div>
        ) : node !== null ? (
          <>
            <div className="progress-meta">
              {history.length > 0 && (
                <Button
                  variant="ghost"
                  className="back-button"
                  onClick={goBack}
                  disabled={isSubmitting}
                  aria-label="Go to previous question"
                >
                  <ArrowLeft size={16} />
                  Back
                </Button>
              )}
              <span className="layer-count">Layer {layer} of 4</span>
            </div>
            <div
              className="progress-track"
              role="progressbar"
              aria-label="Assessment progress"
              aria-valuemin={1}
              aria-valuemax={4}
              aria-valuenow={layer}
            >
              <span style={{ width: `${(layer / 4) * 100}%` }} />
            </div>

            <div className="question-area">
              {node.type === 'screening' ? (
                <>
                  <h1>Do any of these warning signs apply to you?</h1>
                  <div className="red-flag-list" aria-label="Warning signs">
                    {node.redFlags.map((flag) => (
                      <button
                        className={`red-flag-option${selectedRedFlag === flag.id ? ' is-selected' : ''}`}
                        type="button"
                        key={flag.id}
                        onClick={() => setSelectedRedFlag(flag.id)}
                        disabled={isSubmitting}
                        aria-pressed={selectedRedFlag === flag.id}
                      >
                        <span className="selection-indicator" aria-hidden="true" />
                        <span>{flag.text}</span>
                      </button>
                    ))}
                    <button
                      className={`red-flag-option none-option${selectedRedFlag === 'none' ? ' is-selected' : ''}`}
                      type="button"
                      onClick={() => setSelectedRedFlag('none')}
                      disabled={isSubmitting}
                      aria-pressed={selectedRedFlag === 'none'}
                    >
                      <span className="selection-indicator" aria-hidden="true" />
                      <span>None of these apply</span>
                    </button>
                  </div>
                  <Button
                    className="continue-button"
                    onClick={() => selectedRedFlag !== null && void followAnswer(selectedRedFlag)}
                    disabled={selectedRedFlag === null || isSubmitting}
                  >
                    {isSubmitting ? 'Continuing' : 'Continue'}
                    <ArrowRight size={17} />
                  </Button>
                </>
              ) : node.type === 'question' ? (
                <>
                  <h1>{node.questionText}</h1>
                  <div className="answer-list">
                    {node.options.map((option) => (
                      <Button
                        className="answer-button"
                        variant="outline"
                        key={option.value}
                        onClick={() => void followAnswer(option.value)}
                        disabled={isSubmitting}
                      >
                        <span>{option.label}</span>
                        <ArrowRight size={17} aria-hidden="true" />
                      </Button>
                    ))}
                  </div>
                </>
              ) : node.type === 'questionnaire_eval' ? (
                (() => {
                  const isFabq = Number(node.id) < 4000
                  const items = isFabq ? FABQ_ITEMS : QUICKDASH_ITEMS
                  const minimumRating = isFabq ? 0 : 1
                  const maximumRating = isFabq ? 6 : 5
                  const allItemsRated = items.every((item) => formAnswers[item.id] !== undefined)

                  return (
                    <>
                      <h1>{node.title}</h1>
                      <p className="questionnaire-instructions mt-5 leading-7 text-[#555]">
                        {node.questionText}
                      </p>
                      <div className="questionnaire-list mt-7 grid gap-4">
                        {isFabq ? (
                          items.map((item) => (
                            <fieldset
                              className="questionnaire-item grid gap-3 rounded-2xl border border-[#dedede] p-4"
                              key={item.id}
                            >
                              <legend className="px-1 text-sm leading-6">
                                {item.id}. {item.text}
                              </legend>
                              <div className="questionnaire-ratings flex flex-wrap gap-2">
                                {Array.from(
                                  { length: maximumRating - minimumRating + 1 },
                                  (_, index) => minimumRating + index,
                                ).map((rating) => (
                                  <Button
                                    className="questionnaire-rating size-10 px-0"
                                    variant={formAnswers[item.id] === rating ? 'default' : 'outline'}
                                    key={rating}
                                    onClick={() =>
                                      setFormAnswers((answers) => ({ ...answers, [item.id]: rating }))
                                    }
                                    disabled={isSubmitting}
                                    aria-pressed={formAnswers[item.id] === rating}
                                    aria-label={`Rate item ${item.id} ${rating}`}
                                  >
                                    {rating}
                                  </Button>
                                ))}
                              </div>
                            </fieldset>
                          ))
                        ) : (
                          <QuickDashQuestionnaire
                            answers={formAnswers}
                            disabled={isSubmitting}
                            onAnswerChange={(itemId, value) =>
                              setFormAnswers((answers) => ({ ...answers, [itemId]: value }))
                            }
                          />
                        )}
                      </div>
                      <Button
                        className="continue-button"
                        onClick={() =>
                          void followAnswer(
                            undefined,
                            isFabq ? formAnswers : undefined,
                            isFabq ? undefined : formAnswers,
                          )
                        }
                        disabled={!allItemsRated || isSubmitting}
                      >
                        {isSubmitting ? 'Submitting' : 'Submit Questionnaire'}
                        <ArrowRight size={17} />
                      </Button>
                    </>
                  )
                })()
              ) : null}
              {error !== null && (
                <div className="error-state" role="alert">
                  <p>{error}</p>
                  <Button variant="ghost" className="retry-button" onClick={() => void restart()}>
                    Try again
                  </Button>
                </div>
              )}
            </div>
          </>
        ) : (
          <div className="state-message" role="alert">
            <p>{error ?? 'Assessment could not be loaded.'}</p>
            <Button className="restart-button" onClick={restart}>
              Try again
            </Button>
          </div>
        )}
      </section>
    </main>
  )
}

export default App
