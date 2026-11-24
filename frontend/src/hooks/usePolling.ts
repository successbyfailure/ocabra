import { useEffect, useRef } from "react"

export interface PollingOptions {
  /** Si es false, no se programa nada (p. ej. mientras no hay sesión). */
  enabled?: boolean
  /** Ejecutar una vez al montar / al cambiar deps. Por defecto true. */
  immediate?: boolean
  /** Dependencias que, al cambiar, reinician el ciclo (refresca y reprograma).
   *  Útil para refrescar cuando cambian los filtros de la página. */
  deps?: readonly unknown[]
}

/**
 * Sondeo periódico que **se pausa con la pestaña oculta** (``document.hidden``)
 * y se reanuda —con un refresco inmediato— al volver a ella. Antes cada página
 * montaba su propio ``setInterval`` que seguía pegándole al servidor con la
 * pestaña en segundo plano (0 usos de ``visibilitychange`` en todo el front).
 *
 * El callback se guarda en una ref, así que puede cerrar sobre estado reciente
 * sin reprogramar el intervalo; para forzar un refresco al cambiar un filtro,
 * pásalo en ``deps``.
 */
export function usePolling(
  callback: () => void | Promise<void>,
  intervalMs: number,
  options: PollingOptions = {},
): void {
  const { enabled = true, immediate = true, deps = [] } = options
  const savedCallback = useRef(callback)
  savedCallback.current = callback

  useEffect(() => {
    if (!enabled) return
    let timer: number | null = null

    const run = () => {
      void savedCallback.current()
    }
    const start = () => {
      if (timer === null) timer = window.setInterval(run, intervalMs)
    }
    const stop = () => {
      if (timer !== null) {
        window.clearInterval(timer)
        timer = null
      }
    }
    const onVisibility = () => {
      if (document.hidden) {
        stop()
      } else {
        run() // ponerse al día en cuanto el usuario vuelve
        start()
      }
    }

    if (immediate && !document.hidden) run()
    if (!document.hidden) start()
    document.addEventListener("visibilitychange", onVisibility)
    return () => {
      stop()
      document.removeEventListener("visibilitychange", onVisibility)
    }
    // deps controla cuándo reiniciar el ciclo; el resto son estables.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, intervalMs, ...deps])
}
