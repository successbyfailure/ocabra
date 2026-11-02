import type { ReactNode } from "react";

export function FieldHint({ children }: { children: ReactNode }) {
  return <p className="mt-1 text-xs text-muted-foreground">{children}</p>;
}

export function FieldSection({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section className="space-y-4 rounded-lg border border-border/70 bg-background/40 p-4">
      <div>
        <h3 className="text-sm font-medium">{title}</h3>
        <p className="text-xs text-muted-foreground">{description}</p>
      </div>
      {children}
    </section>
  );
}

// Shown inside a backend runtime section when a use case owns context/parallelism/KV.
export function UseCaseManagedNotice({ fields }: { fields: string }) {
  return (
    <div className="rounded-md border border-primary/40 bg-primary/5 px-3 py-2 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">
        Gestionado por el caso de uso.
      </span>{" "}
      {fields} se calculan en el{" "}
      <span className="font-medium">Planificador de contexto y VRAM</span>{" "}
      (arriba); los campos de abajo quedan de solo lectura. Borra el caso de uso
      para editarlos a mano.
    </div>
  );
}

// Visual grouping for a cluster of related fields inside a larger panel —
// the "rounded-md border bg-muted/20" idiom that was repeated ad-hoc
// throughout ModelConfigModal.tsx (Vision, KV cache) before being extracted.
export function SubBlock({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-3 rounded-md border border-border/50 bg-muted/20 p-3">
      <div>
        <h4 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{title}</h4>
        {description && <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>}
      </div>
      {children}
    </div>
  );
}
