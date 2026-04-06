import { useAppState } from "@/lib/state";
import type { ConnectionStatus as ConnectionStatusType } from "@/hooks/useWebSocket";

// ---------------------------------------------------------------------------
// Visual config per connection state
// ---------------------------------------------------------------------------

interface StatusConfig {
  color: string;
  label: string;
}

const STATUS_MAP: Record<ConnectionStatusType, StatusConfig> = {
  connected: { color: "bg-green-500", label: "Connected" },
  disconnected: { color: "bg-red-500", label: "Disconnected" },
  connecting: { color: "bg-yellow-500", label: "Connecting…" },
  reconnecting: { color: "bg-yellow-500", label: "Reconnecting…" },
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ConnectionStatus() {
  const { state } = useAppState();
  const status = state.connectionStatus;
  const { color, label } = STATUS_MAP[status];

  return (
    <div
      data-testid="connection-status"
      className="flex items-center gap-2 text-sm"
      role="status"
      aria-live="polite"
      aria-label={`WebSocket connection: ${label}`}
    >
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${color}`}
        aria-hidden="true"
      />
      <span className="text-muted-foreground">{label}</span>
    </div>
  );
}
