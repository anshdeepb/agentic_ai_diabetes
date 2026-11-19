import { Activity } from "lucide-react"
import { cn } from "@/lib/utils"

export default function BrandMark({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex size-7 shrink-0 items-center justify-center rounded-full border border-border bg-card text-foreground",
        className
      )}
      aria-hidden
    >
      <Activity className="size-3.5" strokeWidth={2.25} />
    </div>
  )
}
