import { useEffect, useState } from 'react'
import { Activity, ArrowLeft, ArrowRight, RotateCcw } from 'lucide-react'
import { Button } from '../components/ui/button'
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

interface DiagnosisNode {
  id: string
  type: 'diagnosis'
  title: string
  primaryDiagnosis: string
  differentials: string[]
  confidenceScore: number
  icdCode?: string | null
}

type DecisionNode = ScreeningNode | QuestionNode | DiagnosisNode

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

  async function followAnswer(answer: string) {
    if (node === null) return

    setIsSubmitting(true)
    setError(null)

    try {
      let result = await requestDecisionTree({ nodeId: node.id, answer })
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
        return
      }

      if (nextNode === null) {
        throw new Error('This path has no next step. Restart the assessment.')
      }

      setHistory((steps) => [...steps, node])
      setNode(nextNode)
      setSelectedRedFlag(null)
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
