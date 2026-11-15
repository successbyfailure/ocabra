import { describe, expect, it } from "vitest";
import { toFederationPeer } from "@/api/client";

describe("toFederationPeer", () => {
  it("reads the peer id from `id`, which is what the API returns", () => {
    // Before 0.82 only peer_id/peerId were read, so every peer had id "" and
    // edit/delete/test from the UI called /peers/ and failed.
    const peer = toFederationPeer({ id: "abc-123", name: "mapache", url: "http://x" });
    expect(peer.peer_id).toBe("abc-123");
  });

  it("maps tunnel and join fields", () => {
    const peer = toFederationPeer({
      id: "p1",
      name: "nodo-b",
      url: "",
      connection_mode: "tunnel_in",
      mode: "swarm",
      local_access_level: "full",
      join_state: null,
      tunnel_connected: true,
      tunnel: {
        remote_node_name: "nodo-b",
        remote_version: "0.82.0",
        connected_at: 1700000000,
        active_streams: 2,
        requests_sent: 10,
        requests_served: 4,
      },
    });
    expect(peer.connection_mode).toBe("tunnel_in");
    expect(peer.mode).toBe("swarm");
    expect(peer.tunnel_connected).toBe(true);
    expect(peer.tunnel?.requests_served).toBe(4);
    expect(peer.tunnel?.remote_node_id).toBeNull();
  });

  it("defaults old peers to one-way HTTP", () => {
    const peer = toFederationPeer({ id: "p2", name: "viejo", url: "http://10.1.0.182:8484" });
    expect(peer.connection_mode).toBe("http");
    expect(peer.mode).toBe("manual");
    expect(peer.tunnel).toBeNull();
    expect(peer.join_state).toBeNull();
  });
});
