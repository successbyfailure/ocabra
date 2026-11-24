// Barrel de tipos (REF-15).
//
// types/index.ts se troceó por dominios (types/models.ts, types/federation.ts,
// …) siguiendo el patrón de types/backends.ts y types/agents.ts. Este fichero
// reexporta todos los tipos para no tocar ninguna importación desde "@/types".
// (types/agents.ts se importa directamente como "@/types/agents".)

export * from "./chat"
export * from "./gpus"
export * from "./models"
export * from "./registry"
export * from "./stats"
export * from "./services"
export * from "./system"
export * from "./auth"
export * from "./federation"
export * from "./batches"
export * from "./benchmarks"
export * from "./backends"
