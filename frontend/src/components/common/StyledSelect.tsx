import * as Select from "@radix-ui/react-select"
import { Check, ChevronDown } from "lucide-react"

export interface SelectOption {
  value: string
  label: string
  disabled?: boolean
}

interface StyledSelectProps {
  value: string
  onValueChange: (value: string) => void
  options: SelectOption[]
  placeholder?: string
  className?: string
  disabled?: boolean
}

// Radix Select does not support empty-string values.
// Use a sentinel internally and convert at the boundary.
const EMPTY_SENTINEL = "__empty__"

function toInternal(v: string): string {
  return v === "" ? EMPTY_SENTINEL : v
}

function toExternal(v: string): string {
  return v === EMPTY_SENTINEL ? "" : v
}

export function StyledSelect({
  value,
  onValueChange,
  options,
  placeholder = "Seleccionar...",
  className = "",
  disabled,
}: StyledSelectProps) {
  const internalOptions = options.map((opt) => ({
    ...opt,
    value: toInternal(opt.value),
  }))

  return (
    <Select.Root
      value={toInternal(value)}
      onValueChange={(v) => onValueChange(toExternal(v))}
      disabled={disabled}
    >
      <Select.Trigger
        className={`inline-flex items-center justify-between gap-2 rounded-md border border-border bg-background px-3 py-2 text-sm text-foreground transition-colors hover:bg-muted/50 focus:outline-none focus:ring-2 focus:ring-primary/50 disabled:opacity-50 data-[placeholder]:text-muted-foreground ${className}`}
      >
        <Select.Value placeholder={placeholder} />
        <Select.Icon>
          <ChevronDown size={14} className="text-muted-foreground" />
        </Select.Icon>
      </Select.Trigger>

      <Select.Portal>
        <Select.Content
          className="z-50 max-h-[300px] min-w-[180px] overflow-hidden rounded-md border border-border bg-card shadow-md"
          position="popper"
          sideOffset={4}
        >
          <Select.Viewport className="p-1">
            {internalOptions.map((opt) => (
              <Select.Item
                key={opt.value}
                value={opt.value}
                disabled={opt.disabled}
                className="flex items-center gap-2 rounded-sm px-2 py-1.5 text-sm cursor-pointer outline-none select-none data-[highlighted]:bg-muted data-[disabled]:opacity-50 data-[disabled]:pointer-events-none"
              >
                <Select.ItemIndicator className="w-4">
                  <Check size={12} />
                </Select.ItemIndicator>
                <Select.ItemText>{opt.label}</Select.ItemText>
              </Select.Item>
            ))}
          </Select.Viewport>
        </Select.Content>
      </Select.Portal>
    </Select.Root>
  )
}
