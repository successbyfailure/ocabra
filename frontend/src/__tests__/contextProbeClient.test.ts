import { describe, expect, it } from "vitest";
import { toContextProbeJob } from "@/api/client";

describe("toContextProbeJob", () => {
  it("maps a finished probe from the API's snake_case", () => {
    const job = toContextProbeJob({
      job_id: "abc",
      profile_id: "qwen3.8:27b",
      preset_key: "normal",
      status: "done",
      gpu_indices: [1],
      context_is_pool: true,
      parallelism: 4,
      steps: [
        { context: 32768, used_mb: { "1": 16544 }, fits: true, note: "" },
        { context: 524288, used_mb: {}, fits: false, note: "did not fit" },
      ],
      evicted: ["whisper/turbo"],
      error: null,
      started_at: 1,
      finished_at: 2,
      result: {
        max_context: 397312,
        used_mb: { "1": 22596 },
        limit_mb: { "1": 23600 },
        kv_mb_per_1k: { "1": 17 },
        capped_by: "memory",
        trained_context: 262144,
        margin_mb: 512,
        full_validated: false,
        vision: {
          projector: "mmproj.gguf",
          cost_mb: { "1": 930 },
          max_context_with_vision: 339968,
        },
      },
    });

    expect(job.status).toBe("done");
    expect(job.steps[1]).toEqual({
      context: 524288,
      usedMb: {},
      fits: false,
      note: "did not fit",
    });
    expect(job.result?.maxContext).toBe(397312);
    expect(job.result?.kvMbPer1k).toEqual({ "1": 17 });
    expect(job.result?.vision?.maxContextWithVision).toBe(339968);
  });

  it("a running probe has no result yet", () => {
    expect(toContextProbeJob({ status: "probing", steps: [] }).result).toBeNull();
  });
});
