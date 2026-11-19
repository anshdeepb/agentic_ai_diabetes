import { Moon, SquarePen, Sun } from "lucide-react"
import { Button } from "@/components/ui/button"
import BrandMark from "@/components/chat/BrandMark"
import { PAGE_GUTTER } from "@/lib/layout"
import type { Theme } from "@/lib/theme"
import { cn } from "@/lib/utils"

interface AppHeaderProps {
  theme: Theme
  onToggleTheme: () => void
  onNewChat: () => void
  canStartNewChat: boolean
}

export default function AppHeader({ theme, onToggleTheme, onNewChat, canStartNewChat }: AppHeaderProps) {
  return (
    <header className="sticky top-0 z-10 bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className={cn("flex h-14 w-full items-center justify-between", PAGE_GUTTER)}>
        <div className="flex items-center gap-2.5">
          <BrandMark />
          <span className="text-sm font-medium tracking-tight">Diabetes Risk Assistant</span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          >
            {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
          </Button>
          <Button variant="ghost" size="sm" onClick={onNewChat} disabled={!canStartNewChat} className="gap-1.5">
            <SquarePen className="size-4" />
            <span className="hidden sm:inline">New chat</span>
          </Button>
        </div>
      </div>
    </header>
  )
}
