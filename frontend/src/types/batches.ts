// Tipos del dominio «batches» (REF-15).
// Extraídos de types/index.ts; el barrel index.ts los reexporta con export *.

export type BatchEndpoint =
  | "/v1/chat/completions"
  | "/v1/embeddings"
  | "/v1/completions"
  | "/v1/images/generations"
  | "/v1/audio/speech"
  | "/v1/audio/transcriptions"
  | "/v1/audio/generate"
  | "/v1/pooling"
  | "/v1/score"
  | "/v1/rerank"
  | "/v1/classify"
  | "/v1/images/edits";

export type BatchStatus =
  | "validating"
  | "in_progress"
  | "finalizing"
  | "completed"
  | "failed"
  | "expired"
  | "cancelling"
  | "cancelled";

export interface OpenAIBatchJob {
  id: string;
  object: "batch";
  endpoint: BatchEndpoint;
  errors: unknown;
  input_file_id: string;
  completion_window: string;
  status: BatchStatus;
  output_file_id: string | null;
  error_file_id: string | null;
  created_at: number | null;
  in_progress_at: number | null;
  expires_at: number | null;
  finalizing_at: number | null;
  completed_at: number | null;
  failed_at: number | null;
  expired_at: number | null;
  cancelling_at: number | null;
  cancelled_at: number | null;
  request_counts: { total: number; completed: number; failed: number };
  total_tokens: number;
  metadata: Record<string, unknown> | null;
  user: { id: string; username: string | null };
}

export interface OpenAIFileObject {
  id: string;
  object: "file";
  bytes: number;
  created_at: number;
  filename: string;
  purpose: string;
  status: string;
  status_details: string | null;
}
