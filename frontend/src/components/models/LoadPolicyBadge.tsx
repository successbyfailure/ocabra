import type { LoadPolicy } from "@/types"

interface LoadPolicyBadgeProps {
  policy: LoadPolicy
}

const LABELS: Record<LoadPolicy, string> = {
  pin: "PIN",
  warm: "WARM",
  on_demand: "ON_DEMAND",
}

export function LoadPolicyBadge({ policy }: LoadPolicyBadgeProps) {
  let classes = "bg-slate-500/20 text-slate-200"

  if (policy === "pin") {
    classes = "bg-violet-500/20 text-violet-200"
  } else if (policy === "warm") {
    classes = "bg-blue-500/20 text-blue-200"
  }

  return (
    <span className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium ${classes}`}>
      {LABELS[policy]}
    </span>
  )
}
