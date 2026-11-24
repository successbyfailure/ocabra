import { describe, expect, it } from "vitest"
import {
  extractOpenAIResponseContent,
  extractToolCalls,
  isEmptyResponseContent,
  mergeToolCalls,
  mergeToolResult,
  parseSseEvent,
  parseToolResultEvent,
  readOllamaChatStream,
  readOpenAIChatStream,
  type AssistantPatch,
} from "@/lib/chatStream"

function sseBody(chunks: string[]): Response {
  const encoder = new TextEncoder()
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk))
      controller.close()
    },
  })
  return { body } as unknown as Response
}

function jsonChunk(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`
}

describe("parseSseEvent", () => {
  it("defaults the event name to message and joins multi-line data", () => {
    expect(parseSseEvent("data: hello\ndata: world")).toEqual({
      name: "message",
      data: "hello\nworld",
    })
  })

  it("reads a named event", () => {
    expect(parseSseEvent("event: ocabra.tool_result\ndata: {}")).toEqual({
      name: "ocabra.tool_result",
      data: "{}",
    })
  })

  it("returns null when there is no data line", () => {
    expect(parseSseEvent("event: ping")).toBeNull()
  })
})

describe("readOpenAIChatStream", () => {
  it("accumulates content and reasoning across events and stops at [DONE]", async () => {
    const patches: AssistantPatch[] = []
    const result = await readOpenAIChatStream(
      sseBody([
        jsonChunk({ choices: [{ delta: { reasoning_content: "pienso" } }] }),
        jsonChunk({ choices: [{ delta: { content: "hola" } }] }),
        jsonChunk({ choices: [{ delta: { content: " mundo" } }] }),
        "data: [DONE]\n\n",
        jsonChunk({ choices: [{ delta: { content: "IGNORADO" } }] }),
      ]),
      (patch) => patches.push(patch),
    )
    expect(result.displayContent).toBe("hola mundo")
    expect(result.reasoning).toBe("pienso")
    expect(patches).toHaveLength(3)
  })

  it("reassembles an event split across read() chunks", async () => {
    const full = jsonChunk({ choices: [{ delta: { content: "partido" } }] })
    const mid = Math.floor(full.length / 2)
    const result = await readOpenAIChatStream(
      sseBody([full.slice(0, mid), full.slice(mid), "data: [DONE]\n\n"]),
      () => {},
    )
    expect(result.displayContent).toBe("partido")
  })

  it("surfaces an error payload embedded in the stream", async () => {
    await expect(
      readOpenAIChatStream(
        sseBody([jsonChunk({ error: { message: "boom" } })]),
        () => {},
      ),
    ).rejects.toThrow("boom")
  })

  it("merges a tool_result event onto the matching tool call", async () => {
    const result = await readOpenAIChatStream(
      sseBody([
        jsonChunk({
          choices: [
            {
              delta: {
                tool_calls: [{ id: "c1", function: { name: "search", arguments: "{}" } }],
              },
            },
          ],
        }),
        `event: ocabra.tool_result\ndata: ${JSON.stringify({
          tool_call_id: "c1",
          tool_name: "search",
          status: "ok",
          duration_ms: 5,
        })}\n\n`,
        "data: [DONE]\n\n",
      ]),
      () => {},
    )
    expect(result.toolCalls).toHaveLength(1)
    expect(result.toolCalls[0].status).toBe("ok")
    expect(result.toolCalls[0].durationMs).toBe(5)
  })
})

describe("readOllamaChatStream", () => {
  it("accumulates NDJSON content lines and stops on done", async () => {
    const result = await readOllamaChatStream(
      sseBody([
        `${JSON.stringify({ message: { content: "ho" } })}\n`,
        `${JSON.stringify({ message: { content: "la" } })}\n`,
        `${JSON.stringify({ done: true })}\n`,
        `${JSON.stringify({ message: { content: "IGNORADO" } })}\n`,
      ]),
      () => {},
    )
    expect(result).toBe("hola")
  })

  it("throws on an error line", async () => {
    await expect(
      readOllamaChatStream(sseBody([`${JSON.stringify({ error: "nope" })}\n`]), () => {}),
    ).rejects.toThrow("nope")
  })
})

describe("extractToolCalls / parseToolResultEvent", () => {
  it("infers subagent kind for delegate_ tools and serialises object args", () => {
    const calls = extractToolCalls([
      { id: "x", function: { name: "delegate_task", arguments: { a: 1 } } },
    ])
    expect(calls[0].kind).toBe("subagent")
    expect(calls[0].args).toContain("\"a\": 1")
  })

  it("returns null when the tool_result event has no id", () => {
    expect(parseToolResultEvent(JSON.stringify({ status: "ok" }))).toBeNull()
  })

  it("appends a tool_result with no prior call as a new entry", () => {
    const patch = parseToolResultEvent(
      JSON.stringify({ tool_call_id: "z", tool_name: "run", status: "timeout" }),
    )!
    const merged = mergeToolResult([], patch)
    expect(merged).toHaveLength(1)
    expect(merged[0].status).toBe("timeout")
  })
})

describe("mergeToolCalls", () => {
  it("merges progressive argument deltas for the same id", () => {
    const first = extractToolCalls([{ id: "a", function: { name: "f", arguments: "{\"q\":" } }])
    const second = extractToolCalls([{ id: "a", function: { name: "f", arguments: "{\"q\":1}" } }])
    const merged = mergeToolCalls(first, second)
    expect(merged).toHaveLength(1)
    expect(merged[0].args).toBe("{\"q\":1}")
  })
})

describe("extractOpenAIResponseContent / isEmptyResponseContent", () => {
  it("extracts the non-streaming message body", () => {
    const content = extractOpenAIResponseContent({
      choices: [{ message: { content: "final" } }],
    })
    expect(content.displayContent).toBe("final")
    expect(isEmptyResponseContent(content)).toBe(false)
  })

  it("flags an all-empty response", () => {
    expect(
      isEmptyResponseContent({ displayContent: "", apiContent: "", reasoning: "", toolCalls: [] }),
    ).toBe(true)
  })
})
