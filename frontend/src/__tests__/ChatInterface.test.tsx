// @vitest-environment jsdom
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ChatInterface } from "@/components/playground/ChatInterface"

const params = {
  temperature: 0.7,
  maxTokens: 256,
  topP: 0.9,
  systemPrompt: "system prompt",
  responseFormat: "text" as const,
}

describe("ChatInterface", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("does not send reasoning fallback text back to agent history on the second turn", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([
          jsonChunk({
            choices: [
              {
                delta: {
                  role: "assistant",
                  reasoning_content: "razonamiento visible",
                },
              },
            ],
          }),
          jsonChunk({
            choices: [{ delta: {}, finish_reason: "stop" }],
          }),
          "data: [DONE]\n\n",
        ]),
      )
      .mockResolvedValueOnce(
        sseResponse([
          jsonChunk({
            choices: [
              {
                delta: {
                  role: "assistant",
                  content: "respuesta final",
                },
              },
            ],
          }),
          jsonChunk({
            choices: [{ delta: {}, finish_reason: "stop" }],
          }),
          "data: [DONE]\n\n",
        ]),
      )

    vi.stubGlobal("fetch", fetchMock)

    render(<ChatInterface modelId="agent/glados-mks" backendType="vllm" params={params} />)

    const input = screen.getByPlaceholderText("Escribe tu mensaje...")
    const sendButton = screen.getByRole("button", { name: /enviar/i })

    fireEvent.change(input, { target: { value: "primer turno" } })
    fireEvent.click(sendButton)

    await waitFor(() => {
      expect(screen.getByText("razonamiento visible")).toBeTruthy()
    })

    fireEvent.change(input, { target: { value: "segundo turno" } })
    fireEvent.click(sendButton)

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2)
    })

    const secondRequest = fetchMock.mock.calls[1]?.[1]
    const body = JSON.parse(String(secondRequest?.body)) as {
      messages: Array<{ role: string; content: unknown }>
    }
    const assistantMessage = body.messages.find((message) => message.role === "assistant")

    expect(assistantMessage).toBeTruthy()
    expect(assistantMessage?.content).toBe("")

    await waitFor(() => {
      expect(screen.getByText("respuesta final")).toBeTruthy()
    })
  })

  it("renders thought, tool calls and subagent activity from the agent stream", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        jsonChunk({
          choices: [
            {
              delta: {
                role: "assistant",
                reasoning_content: "voy a delegar este trabajo",
                tool_calls: [
                  {
                    id: "call-1",
                    function: {
                      name: "delegate_task_bot",
                      arguments: "{\"task\":\"resume esto\"}",
                    },
                  },
                ],
              },
            },
          ],
        }),
        eventChunk(
          "ocabra.tool_result",
          JSON.stringify({
            tool_call_id: "call-1",
            alias: "agent",
            tool_name: "task-bot",
            status: "ok",
            duration_ms: 12,
            error: null,
          }),
        ),
        jsonChunk({
          choices: [
            {
              delta: {
                content: "resultado final",
              },
            },
          ],
        }),
        jsonChunk({
          choices: [{ delta: {}, finish_reason: "stop" }],
        }),
        "data: [DONE]\n\n",
      ]),
    )

    vi.stubGlobal("fetch", fetchMock)

    render(<ChatInterface modelId="agent/glados-mks" backendType="vllm" params={params} />)

    fireEvent.change(screen.getByPlaceholderText("Escribe tu mensaje..."), {
      target: { value: "hazlo" },
    })
    fireEvent.click(screen.getByRole("button", { name: /enviar/i }))

    await waitFor(() => {
      expect(screen.getByText("Pensamiento")).toBeTruthy()
      expect(screen.getByText("Tools y subagentes")).toBeTruthy()
      expect(screen.getByText("agent/task-bot")).toBeTruthy()
      expect(screen.getByText("resultado final")).toBeTruthy()
    })
  })

  it("accepts text-like reasoning payloads instead of leaving the message empty", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      sseResponse([
        jsonChunk({
          choices: [
            {
              delta: {
                role: "assistant",
                reasoning_content: [{ type: "text", text: "razonamiento estructurado" }],
              },
            },
          ],
        }),
        jsonChunk({
          choices: [{ delta: {}, finish_reason: "stop" }],
        }),
        "data: [DONE]\n\n",
      ]),
    )

    vi.stubGlobal("fetch", fetchMock)

    render(<ChatInterface modelId="agent/glados-mks" backendType="vllm" params={params} />)

    fireEvent.change(screen.getByPlaceholderText("Escribe tu mensaje..."), {
      target: { value: "piensa en voz alta" },
    })
    fireEvent.click(screen.getByRole("button", { name: /enviar/i }))

    await waitFor(() => {
      expect(screen.getByText("razonamiento estructurado")).toBeTruthy()
    })
  })

  it("retries in non-streaming mode when the agent stream finishes completely empty", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(
        sseResponse([
          jsonChunk({
            choices: [{ delta: { role: "assistant", content: "" }, finish_reason: null }],
          }),
          jsonChunk({
            choices: [{ delta: {}, finish_reason: "stop" }],
          }),
          "data: [DONE]\n\n",
        ]),
      )
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          choices: [
            {
              message: {
                content: "fallback final",
              },
            },
          ],
        }),
      })

    vi.stubGlobal("fetch", fetchMock)

    render(<ChatInterface modelId="agent/glados-mks" backendType="vllm" params={params} />)

    fireEvent.change(screen.getByPlaceholderText("Escribe tu mensaje..."), {
      target: { value: "responde algo" },
    })
    fireEvent.click(screen.getByRole("button", { name: /enviar/i }))

    await waitFor(() => {
      expect(screen.getByText("fallback final")).toBeTruthy()
      expect(fetchMock).toHaveBeenCalledTimes(2)
    })
  })
})

function sseResponse(events: string[]) {
  const encoder = new TextEncoder()
  const body = new ReadableStream({
    start(controller) {
      for (const event of events) {
        controller.enqueue(encoder.encode(event))
      }
      controller.close()
    },
  })
  return { ok: true, body }
}

function jsonChunk(payload: unknown): string {
  return `data: ${JSON.stringify(payload)}\n\n`
}

function eventChunk(name: string, payload: string): string {
  return `event: ${name}\ndata: ${payload}\n\n`
}
