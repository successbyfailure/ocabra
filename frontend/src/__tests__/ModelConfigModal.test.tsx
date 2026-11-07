// @vitest-environment jsdom
import { act, fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { api } from "@/api/client"
import { ModelConfigModal } from "@/components/models/ModelConfigModal"
import type { GPUState, LlamaCppConfig, ModelState } from "@/types"

vi.mock("@/api/client", () => ({
  api: {
    models: {
      estimateMemory: vi.fn(async () => ({
        backendType: "vllm",
        gpuIndex: 1,
        totalVramMb: 24576,
        freeVramMb: 24115,
        budgetVramMb: 22118,
        requestedContextLength: 7800,
        estimatedWeightsMb: 18000,
        estimatedEngineMbPerGpu: null,
        estimatedKvCacheMb: 1956,
        estimatedMaxContextLength: 7824,
        modelLoadingMemoryMb: 18575,
        maximumConcurrency: 1,
        tensorParallelSize: 1,
        fitsCurrentGpu: true,
        enginePresent: null,
        source: "heuristic",
        status: "ok",
        warning: null,
        notes: ["estimacion de prueba"],
      })),
    },
  },
}))

describe("ModelConfigModal", () => {
  it("renders persisted recipe metadata and reapplies suggested config", () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "Qwen/Qwen3-8B-Instruct",
          displayName: "Qwen3 8B",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "warm",
          pinPermanent: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          ramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
    imageEditing: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
            streaming: true,
            contextLength: 32768,
            nativeContextLength: 32768,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              recipeId: "qwen3",
              recipeNotes: ["Qwen3 necesita parser de tools y reasoning."],
              recipeModelImpl: "vllm",
              recipeRunner: "generate",
              suggestedConfig: {
                tool_call_parser: "qwen3_json",
                reasoning_parser: "qwen3",
              },
              suggestedTuning: {
                gpu_memory_utilization: 0.9,
                enable_prefix_caching: true,
              },
              probeStatus: "supported_transformers_backend",
              probeRecommendedModelImpl: "transformers",
              probeRecommendedRunner: "generate",
              probeReason: "Compatibilidad verificada via Transformers backend.",
              probeObservedAt: "2026-03-14T12:00:00Z",
              modelImpl: "vllm",
              runner: "generate",
            },
          },
        }}
      />,
    )

    expect(screen.getByText(/recipe=qwen3/i)).toBeTruthy()
    expect(screen.getByText(/recipe=qwen3 · model_impl=vllm · runner=generate/i)).toBeTruthy()
    expect(screen.getByText(/Qwen3 necesita parser/i)).toBeTruthy()
    expect(screen.getByText(/probe=compatibilidad verificada via Transformers backend/i)).toBeTruthy()
    expect(screen.getByText(/recomendacion verificada · model_impl=transformers · runner=generate/i)).toBeTruthy()
    expect(screen.getByText(/difiere de la recomendacion verificada por probe/i)).toBeTruthy()
    expect(screen.getByText(/La recipe base y la recomendacion final del probe no coinciden/i)).toBeTruthy()
    expect(screen.getByText(/observado=2026-03-14T12:00:00Z/i)).toBeTruthy()

    fireEvent.click(screen.getByText(/Reaplicar sugerencias de recipe/i))

    expect(screen.getByDisplayValue("qwen3_json")).toBeTruthy()
    expect(screen.getByDisplayValue("qwen3")).toBeTruthy()

    fireEvent.click(screen.getByText(/Aplicar tuning recomendado/i))

    expect((screen.getAllByLabelText(/GPU memory utilization/i)[1] as HTMLInputElement).value).toBe("0.9")

    fireEvent.click(screen.getByText(/Aplicar recomendacion del probe/i))

    expect(screen.getByDisplayValue("transformers")).toBeTruthy()
  })

  it("shows override hint when probe flags a missing parser", () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "DeepSeek/DeepSeek-R1",
          displayName: "DeepSeek R1",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "warm",
          pinPermanent: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          ramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
    imageEditing: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
            streaming: true,
            contextLength: 32768,
            nativeContextLength: 32768,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              probeStatus: "missing_reasoning_parser",
              probeRecommendedModelImpl: "vllm",
              probeRecommendedRunner: "generate",
              probeReason: "Reasoning needs parser.",
              modelImpl: "vllm",
              runner: "generate",
            },
          },
        }}
      />,
    )

    expect(screen.getByText(/probe=falta reasoning_parser/i)).toBeTruthy()
    expect(screen.getByText(/override sugerido: reasoning_parser/i)).toBeTruthy()
  })

  it("saves llama.cpp options in nested backend config", async () => {
    const onSave = vi.fn(async () => undefined)

    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={onSave}
        gpus={[]}
        model={{
          modelId: "llama_cpp/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
          displayName: "Qwen2.5 GGUF",
          backendType: "llama_cpp",
          status: "configured",
          loadPolicy: "on_demand",
          pinPermanent: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          ramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: false,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: false,
            imageGeneration: false,
    imageEditing: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
            streaming: true,
            contextLength: 8192,
            nativeContextLength: 8192,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            model_path: "/models/qwen.gguf",
            llama_cpp: {
              mmproj_model_id: "/models/legacy-mmproj.gguf",
              vision_mode: "on_demand",
              kv_unified: true,
              parallel_slots: 4,
            } as unknown as LlamaCppConfig,
          },
        }}
      />,
    )

    expect((screen.getByLabelText(/Projector multimodal/i) as HTMLInputElement).value).toBe(
      "/models/legacy-mmproj.gguf",
    )
    expect((screen.getByLabelText(/Política de visión/i) as HTMLSelectElement).value).toBe(
      "on_demand",
    )
    expect(screen.getByText(/Pool KV total compartido por 4 slots/i)).toBeTruthy()
    expect((screen.getByDisplayValue("activada") as HTMLSelectElement).value).toBe("true")
    fireEvent.change(screen.getByLabelText(/GPU layers/i), { target: { value: "24" } })
    fireEvent.change(screen.getByLabelText(/Context size/i), { target: { value: "16384" } })
    fireEvent.click(screen.getByLabelText(/flash attention/i))
    fireEvent.click(screen.getByLabelText(/embedding mode/i))
    fireEvent.change(screen.getByLabelText(/Projector multimodal/i), {
      target: { value: "/models/mmproj.gguf" },
    })
    fireEvent.change(screen.getByLabelText(/Política de visión/i), {
      target: { value: "on_demand" },
    })
    await act(async () => {
      fireEvent.click(screen.getByText(/^Guardar$/i))
    })

    expect(onSave).toHaveBeenCalledWith(
      "llama_cpp/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
      expect.objectContaining({
        extraConfig: expect.objectContaining({
          model_path: "/models/qwen.gguf",
          // extra_config is stored in snake_case since migration 0034:
          // the modal writes snake_case only, never both spellings.
          llama_cpp: expect.objectContaining({
            gpu_layers: 24,
            ctx_size: 16384,
            flash_attn: true,
            embedding: true,
            mmproj_model_id: "/models/mmproj.gguf",
            vision_mode: "on_demand",
            kv_unified: true,
            parallel_slots: 4,
          }),
        }),
      }),
    )
    const saved = (onSave.mock.calls[0] as unknown as [string, { extraConfig: { llama_cpp: Record<string, unknown> } }])[1]
    const llamaKeys = Object.keys(saved.extraConfig.llama_cpp)
    expect(llamaKeys.filter((key) => /[A-Z]/.test(key))).toEqual([])
  })

  it("renders memory estimate and lets the user trigger runtime validation", async () => {
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={vi.fn(async () => undefined)}
        gpus={[]}
        model={{
          modelId: "vllm/Qwen/Qwen3-32B-AWQ",
          displayName: "Qwen3 32B AWQ",
          backendType: "vllm",
          status: "configured",
          loadPolicy: "on_demand",
          pinPermanent: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          ramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: true,
            completion: true,
            tools: true,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: true,
            imageGeneration: false,
    imageEditing: false,
            audioTranscription: false,
            musicGeneration: false,
            tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
            streaming: true,
            contextLength: 40960,
            nativeContextLength: 40960,
          },
          lastRequestAt: null,
          loadedAt: null,
          extraConfig: {
            vllm: {
              maxModelLen: 7800,
              gpuMemoryUtilization: 0.9,
            },
          },
        }}
      />,
    )

    await screen.findByText(/Estimación por probe \(runtime\)/i)
    await screen.findByText("17.6 GB")
    await screen.findByText("Contexto máximo")
    expect(screen.getAllByText("7.8k").length).toBeGreaterThan(0)

    await act(async () => {
      fireEvent.click(screen.getByText(/Validar con engine/i))
    })

    expect(api.models.estimateMemory).toHaveBeenCalled()
  })
  it("reads snake_case whisper config (diarization) and writes it back in snake_case", async () => {
    const onSave = vi.fn(async () => undefined)
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={onSave}
        gpus={[]}
        model={{
          modelId: "whisper/large-v3",
          displayName: "Whisper large-v3",
          backendType: "whisper",
          status: "configured",
          loadPolicy: "on_demand",
          pinPermanent: false,
          preferredGpu: null,
          currentGpu: [],
          vramUsedMb: 0,
          ramUsedMb: 0,
          diskSizeBytes: null,
          capabilities: {
            chat: false,
            completion: false,
            tools: false,
            vision: false,
            embeddings: false,
            pooling: false,
            rerank: false,
            classification: false,
            score: false,
            reasoning: false,
            imageGeneration: false,
            imageEditing: false,
            audioTranscription: true,
            musicGeneration: false,
            tts: false,
            audioInput: true,
            audioOutput: false,
            videoInput: false,
            streaming: false,
            contextLength: 0,
            nativeContextLength: 0,
          },
          lastRequestAt: null,
          loadedAt: null,
          // Raw server shape (snake_case), plus a stale camelCase twin that
          // must lose against the persisted snake_case value.
          extraConfig: {
            whisper: {
              diarization_enabled: true,
              diarizationEnabled: false,
              diarization_model_id: "pyannote/speaker-diarization-3.1",
              num_workers: 3,
            },
          } as unknown as ModelState["extraConfig"],
        }}
      />,
    )

    const checkbox = screen.getByLabelText(/habilitar diarización/i) as HTMLInputElement
    expect(checkbox.checked).toBe(true)
    expect(screen.getByDisplayValue("pyannote/speaker-diarization-3.1")).toBeTruthy()
    expect(screen.getByDisplayValue("3")).toBeTruthy()

    await act(async () => {
      fireEvent.click(screen.getByText(/^Guardar$/i))
    })

    const saved = (onSave.mock.calls[0] as unknown as [string, { extraConfig: Record<string, unknown> }])[1]
    expect(saved.extraConfig.whisper).toEqual({
      diarization_enabled: true,
      diarization_model_id: "pyannote/speaker-diarization-3.1",
      num_workers: 3,
    })
  })
  function baseModel(overrides: Partial<ModelState>): ModelState {
    return {
      modelId: "x",
      displayName: "x",
      backendType: "llamacpp_mks",
      status: "configured",
      loadPolicy: "on_demand",
      pinPermanent: false,
      preferredGpu: null,
      currentGpu: [],
      vramUsedMb: 0,
      ramUsedMb: 0,
      diskSizeBytes: null,
      capabilities: {
        chat: true,
        completion: true,
        tools: false,
        vision: false,
        embeddings: false,
        pooling: false,
        rerank: false,
        classification: false,
        score: false,
        reasoning: false,
        imageGeneration: false,
        imageEditing: false,
        audioTranscription: false,
        musicGeneration: false,
        tts: false,
        audioInput: false,
        audioOutput: false,
        videoInput: false,
        streaming: true,
        contextLength: 8192,
        nativeContextLength: 8192,
      },
      lastRequestAt: null,
      loadedAt: null,
      ...overrides,
    }
  }

  it("sends null for a stored field the user emptied, and nothing for untouched unset ones", async () => {
    const onSave = vi.fn(async () => undefined)
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={onSave}
        gpus={[]}
        model={baseModel({
          modelId: "llamacpp_mks/qwen3:8b",
          extraConfig: {
            llama_cpp: {
              model_path: "/models/q.gguf",
              ctx_size: 8192,
              gpu_layers: 99,
              threads: 8,
            } as unknown as LlamaCppConfig,
          },
        })}
      />,
    )

    expect((screen.getByLabelText(/Context size/i) as HTMLInputElement).value).toBe("8192")
    fireEvent.change(screen.getByLabelText(/Context size/i), { target: { value: "" } })
    fireEvent.change(screen.getByLabelText(/GPU layers/i), { target: { value: "" } })
    await act(async () => {
      fireEvent.click(screen.getByText(/^Guardar$/i))
    })

    const saved = (onSave.mock.calls[0] as unknown as [string, { extraConfig: { llama_cpp: Record<string, unknown> } }])[1]
    const llama = saved.extraConfig.llama_cpp
    expect(llama.ctx_size).toBeNull()
    expect(llama.gpu_layers).toBeNull()
    expect(llama.threads).toBe(8)
    expect(llama.model_path).toBe("/models/q.gguf")
    // Never stored, never touched: no null for these.
    expect("batch_size" in llama).toBe(false)
    expect("tensor_split" in llama).toBe(false)
    expect("speculative" in llama).toBe(false)
  })

  it("edits whisper.cpp threads and diarization model, with the per-request diarize note", async () => {
    const onSave = vi.fn(async () => undefined)
    render(
      <ModelConfigModal
        open
        onOpenChange={vi.fn()}
        onSave={onSave}
        gpus={[{ index: 1, name: "RTX 3060" } as unknown as GPUState]}
        model={baseModel({
          modelId: "whisper_cpp/large-v3-turbo",
          backendType: "whisper_cpp",
          preferredGpu: 1,
          extraConfig: { whisper_cpp: { threads: 6, model_path: "/m/ggml.bin" } },
        })}
      />,
    )

    // GPU preferida is part of every deployment form.
    expect((screen.getByLabelText(/GPU preferida/i) as HTMLSelectElement).value).toBe("1")
    expect((screen.getByLabelText("whisper.cpp threads") as HTMLInputElement).value).toBe("6")
    expect(screen.getByPlaceholderText("pyannote/speaker-diarization-3.1")).toBeTruthy()
    expect(document.body.textContent).toContain("diarize=true")
    expect(screen.queryByLabelText(/habilitar diarización/i)).toBeNull()

    fireEvent.change(screen.getByLabelText("whisper.cpp threads"), { target: { value: "" } })
    fireEvent.change(screen.getByLabelText("whisper.cpp diarization model id"), {
      target: { value: "pyannote/speaker-diarization-community-1" },
    })
    await act(async () => {
      fireEvent.click(screen.getByText(/^Guardar$/i))
    })

    const saved = (onSave.mock.calls[0] as unknown as [string, { preferredGpu: number | null; extraConfig: Record<string, unknown> }])[1]
    expect(saved.preferredGpu).toBe(1)
    expect(saved.extraConfig.whisper_cpp).toEqual({
      model_path: "/m/ggml.bin",
      threads: null,
      diarization_model_id: "pyannote/speaker-diarization-community-1",
    })
  })
})
