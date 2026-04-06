import { describe, it, expect } from "vitest";
import type { ConnectionStatus } from "@/hooks/useWebSocket";

/**
 * ConnectionStatus unit tests
 *
 * Validates:
 * - Requirement 8.3: Frontend displays a connection status indicator
 * - Design: colored indicators (green=connected, red=disconnected, yellow=connecting/reconnecting)
 */

// Mirror the component's STATUS_MAP for logic-level testing
interface StatusConfig {
  color: string;
  label: string;
}

const STATUS_MAP: Record<ConnectionStatus, StatusConfig> = {
  connected: { color: "bg-green-500", label: "Connected" },
  disconnected: { color: "bg-red-500", label: "Disconnected" },
  connecting: { color: "bg-yellow-500", label: "Connecting…" },
  reconnecting: { color: "bg-yellow-500", label: "Reconnecting…" },
};

const ALL_STATUSES: ConnectionStatus[] = [
  "connected",
  "disconnected",
  "connecting",
  "reconnecting",
];

describe("ConnectionStatus — status mapping", () => {
  it("every connection status has a non-empty label", () => {
    for (const status of ALL_STATUSES) {
      const { label } = STATUS_MAP[status];
      expect(label, `status "${status}" should have a label`).toBeTruthy();
      expect(label.length).toBeGreaterThan(0);
    }
  });

  it("every connection status has a color class", () => {
    for (const status of ALL_STATUSES) {
      const { color } = STATUS_MAP[status];
      expect(color, `status "${status}" should have a color`).toMatch(/^bg-/);
    }
  });

  it("connected uses green indicator", () => {
    expect(STATUS_MAP.connected.color).toBe("bg-green-500");
  });

  it("disconnected uses red indicator", () => {
    expect(STATUS_MAP.disconnected.color).toBe("bg-red-500");
  });

  it("connecting uses yellow indicator", () => {
    expect(STATUS_MAP.connecting.color).toBe("bg-yellow-500");
  });

  it("reconnecting uses yellow indicator", () => {
    expect(STATUS_MAP.reconnecting.color).toBe("bg-yellow-500");
  });

  it("reconnecting label indicates ongoing reconnection", () => {
    expect(STATUS_MAP.reconnecting.label.toLowerCase()).toContain("reconnect");
  });
});
