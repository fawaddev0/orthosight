interface QuickDashItem {
  id: number
  text: string
}

interface QuickDashQuestionnaireProps {
  answers: Record<number, number>
  disabled?: boolean
  onAnswerChange: (itemId: number, value: number) => void
}

const QUICKDASH_ITEMS: QuickDashItem[] = [
  { id: 1, text: 'Open a tight or new jar.' },
  { id: 2, text: 'Do heavy household tasks.' },
  { id: 3, text: 'Reach overhead.' },
  { id: 4, text: 'Wash your back.' },
  { id: 5, text: 'Write.' },
  { id: 6, text: 'Turn a key.' },
  { id: 7, text: 'How severe is your arm, shoulder, or hand pain?' },
  { id: 8, text: 'How much pain do you have during activity?' },
  { id: 9, text: 'How much tingling or numbness do you feel?' },
  { id: 10, text: 'How much does pain disrupt your sleep?' },
  { id: 11, text: 'How much does your condition interfere with social or other activities?' },
]

const ACTIVITY_LABELS = [
  'No difficulty',
  'Mild difficulty',
  'Moderate difficulty',
  'Severe difficulty',
  'Unable to do',
] as const

const SYMPTOM_LABELS = ['None', 'Mild', 'Moderate', 'Severe', 'Extreme'] as const

const IMPACT_LABELS = [
  'Not at all',
  'Slightly',
  'Moderately',
  'Quite a bit',
  'Extremely / Unable',
] as const

function labelsForItem(itemId: number): readonly string[] {
  if (itemId <= 6) return ACTIVITY_LABELS
  if (itemId <= 9) return SYMPTOM_LABELS
  return IMPACT_LABELS
}

export function QuickDashQuestionnaire({
  answers,
  disabled = false,
  onAnswerChange,
}: QuickDashQuestionnaireProps) {
  return (
    <div className="grid gap-4">
      {QUICKDASH_ITEMS.map((item) => (
        <fieldset className="grid gap-3 rounded-xl border border-[#dedede] p-4" key={item.id}>
          <legend className="px-1 text-sm leading-6">
            {item.id}. {item.text}
          </legend>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
            {labelsForItem(item.id).map((label, index) => {
              const value = index + 1
              const isSelected = answers[item.id] === value

              return (
                <button
                  aria-label={`${value} - ${label}`}
                  aria-pressed={isSelected}
                  className={`flex min-h-16 flex-col items-center justify-center gap-1 rounded-lg border px-2 py-2 text-center transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#171717] disabled:cursor-not-allowed disabled:opacity-60 ${
                    isSelected
                      ? 'border-[#171717] bg-[#171717] text-white'
                      : 'border-[#dedede] bg-white text-[#292929] hover:bg-[#f3f3f3]'
                  }`}
                  disabled={disabled}
                  key={value}
                  onClick={() => onAnswerChange(item.id, value)}
                  type="button"
                >
                  <span className="text-sm font-semibold">{value}</span>
                  <span className="text-xs leading-tight">{label}</span>
                </button>
              )
            })}
          </div>
        </fieldset>
      ))}
    </div>
  )
}
