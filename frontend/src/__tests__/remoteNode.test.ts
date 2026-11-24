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
