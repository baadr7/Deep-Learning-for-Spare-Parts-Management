import { afterEach, describe, expect, it, vi } from "vitest";
import { liveSocketUrl } from "./api";

afterEach(() => vi.unstubAllGlobals());

describe("liveSocketUrl", () => {
  it("pointe vers le canal WebSocket de l'hôte courant", () => {
    const url = liveSocketUrl();
    expect(url.endsWith("/api/ws/live")).toBe(true);
    expect(url.startsWith("ws://") || url.startsWith("wss://")).toBe(true);
  });

  it("bascule en ws:// sur une page servie en http", () => {
    vi.stubGlobal("location", { protocol: "http:", host: "localhost:8000" });
    expect(liveSocketUrl()).toBe("ws://localhost:8000/api/ws/live");
  });

  it("bascule en wss:// sur une page servie en https", () => {
    vi.stubGlobal("location", { protocol: "https:", host: "exemple.fr" });
    expect(liveSocketUrl()).toBe("wss://exemple.fr/api/ws/live");
  });
});
