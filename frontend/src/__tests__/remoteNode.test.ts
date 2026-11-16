import { describe, expect, it } from "vitest";
import { remotePath } from "@/lib/remoteNode";

const node = { id: "peer 1", name: "Mapache" };

describe("remotePath", () => {
  it("leaves paths alone when no remote node is selected", () => {
    expect(remotePath("/ocabra/models", null)).toBe("/ocabra/models");
  });

  it("sends the node's API through the panel proxy, query included", () => {
    expect(remotePath("/ocabra/models?include=x", node)).toBe(
      "/ocabra/federation/peers/peer%201/proxy/ocabra/models?include=x",
    );
    expect(remotePath("/v1/models", node)).toBe("/ocabra/federation/peers/peer%201/proxy/v1/models");
    expect(remotePath("/api/tags", node)).toBe("/ocabra/federation/peers/peer%201/proxy/api/tags");
  });

  it("keeps login and the federation/cloud pages on this server", () => {
    for (const p of ["/ocabra/auth/me", "/ocabra/federation/peers", "/ocabra/cloud/nodes"]) {
      expect(remotePath(p, node)).toBe(p);
    }
  });

  it("does not touch non-API paths", () => {
    expect(remotePath("/health", node)).toBe("/health");
  });
});

describe("installRemoteRouting", () => {
  it("sends every fetch and EventSource of the UI to the remote node", async () => {
    const { installRemoteRouting } = await import("@/lib/remoteNode");
    const seen: string[] = [];
    const origFetch = window.fetch;
    const OrigES = window.EventSource;
    window.fetch = ((input: RequestInfo | URL) => {
      seen.push(typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url);
      return Promise.resolve(new Response("{}"));
    }) as typeof fetch;
    class FakeES {
      url: string;
      constructor(url: string | URL) {
        this.url = url.toString();
        seen.push(this.url);
      }
    }
    window.EventSource = FakeES as unknown as typeof EventSource;
    try {
      installRemoteRouting(node);
      await fetch("/ocabra/mcp-servers");
      await fetch(`${window.location.origin}/v1/chat/completions`);
      await fetch(new Request(`${window.location.origin}/ocabra/agents`));
      await fetch("/ocabra/federation/peers/peer%201/proxy/ocabra/models"); // already rewritten
      await fetch("/ocabra/auth/me"); // stays local
      await fetch("https://example.com/x"); // other origins untouched
      new EventSource("/ocabra/downloads/j1/stream");
      const p = "/ocabra/federation/peers/peer%201/proxy";
      expect(seen).toEqual([
        `${p}/ocabra/mcp-servers`,
        `${window.location.origin}${p}/v1/chat/completions`,
        `${window.location.origin}${p}/ocabra/agents`,
        `${p}/ocabra/models`,
        "/ocabra/auth/me",
        "https://example.com/x",
        `${p}/ocabra/downloads/j1/stream`,
      ]);
    } finally {
      window.fetch = origFetch;
      window.EventSource = OrigES;
    }
  });
});
