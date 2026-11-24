// Cliente tipado de /ocabra/* — dominio «batches» (REF-15).
// Extraído de api/client.ts; usa la capa HTTP única de api/http.ts.

import { extractErrorMessage, httpFetch, request } from "@/api/http"
import { BASE, buildQuery } from "@/api/_helpers"
import type {
  BatchEndpoint,
  OpenAIBatchJob,
  OpenAIFileObject,
} from "@/types"

export const batchesApiModule = {
  batches: {
    uploadFile: async (
      file: File,
      purpose: string,
    ): Promise<OpenAIFileObject> => {
      const form = new FormData();
      form.append("file", file);
      form.append("purpose", purpose);
      const res = await httpFetch("/v1/files", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(extractErrorMessage(err) ?? res.statusText);
      }
      return res.json() as Promise<OpenAIFileObject>;
    },
    fileContentUrl: (fileId: string) =>
      `${BASE}/v1/files/${encodeURIComponent(fileId)}/content`,
    create: (payload: {
      inputFileId: string;
      endpoint: BatchEndpoint;
      metadata?: Record<string, unknown>;
    }) =>
      request<OpenAIBatchJob>("POST", `/v1/batches`, {
        input_file_id: payload.inputFileId,
        endpoint: payload.endpoint,
        completion_window: "24h",
        ...(payload.metadata && { metadata: payload.metadata }),
      }),
    list: (params: { after?: string; limit?: number } = {}) => {
      const query = buildQuery({ after: params.after, limit: params.limit });
      return request<{
        object: string;
        data: OpenAIBatchJob[];
        first_id: string | null;
        last_id: string | null;
        has_more: boolean;
      }>("GET", `/v1/batches${query}`);
    },
    cancel: (batchId: string) =>
      request<OpenAIBatchJob>(
        "POST",
        `/v1/batches/${encodeURIComponent(batchId)}/cancel`,
      ),
  },

};
