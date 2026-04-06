import { useCallback, useEffect, useRef, useState } from "react";
import type { WSMessage } from "../types";

/** Connection states exposed to consumers. */
export type ConnectionStatus = "connecting" | "connected" | "disconnected" | "reconnecting";

export interface UseWebSocketOptions {
  /** Full WebSocket URL, e.g. `ws://localhost:8000/ws`. */
  url: string;
  /** Called for every validated message received from the server. */
  onMessage?: (msg: WSMessage) => void;
  /** Whether the hook should attempt to connect automatically. Default `true`. */
  autoConnect?: boolean;
}

export interface UseWebSocketReturn {
  status: ConnectionStatus;
  /** Manually trigger a (re)connection attempt. */
  connect: () => void;
  /** Gracefully close the connection (disables auto-reconnect). */
  disconnect: () => void;
}

// Exponential back-off constants (seconds → ms internally).
const BASE_DELAY_MS = 1_000;
const MAX_DELAY_MS = 30_000;

function nextDelay(attempt: number): number {
  // 1s, 2s, 4s, 8s, … capped at 30s
  return Math.min(BASE_DELAY_MS * 2 ** attempt, MAX_DELAY_MS);
}

/**
 * React hook that manages a single WebSocket connection with automatic
 * reconnection using exponential back-off (1 s → 2 s → 4 s … max 30 s).
 */
export function useWebSocket({
  url,
  onMessage,
  autoConnect = true,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [status, setStatus] = useState<ConnectionStatus>("disconnected");

  const wsRef = useRef<WebSocket | null>(null);
  const attemptRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const intentionalCloseRef = useRef(false);
  // Keep a stable reference to the latest onMessage callback.
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const doConnect = useCallback(() => {
    // Prevent duplicate connections.
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.CONNECTING ||
        wsRef.current.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    intentionalCloseRef.current = false;
    setStatus(attemptRef.current === 0 ? "connecting" : "reconnecting");

    const ws = new WebSocket(url);

    ws.onopen = () => {
      attemptRef.current = 0;
      setStatus("connected");
    };

    ws.onmessage = (event: MessageEvent) => {
      try {
        const data = JSON.parse(event.data as string) as WSMessage;
        onMessageRef.current?.(data);
      } catch {
        // Ignore malformed frames.
      }
    };

    ws.onclose = () => {
      wsRef.current = null;
      setStatus("disconnected");

      if (!intentionalCloseRef.current) {
        const delay = nextDelay(attemptRef.current);
        attemptRef.current += 1;
        setStatus("reconnecting");
        timerRef.current = setTimeout(doConnect, delay);
      }
    };

    ws.onerror = () => {
      // The browser will fire `onclose` after `onerror`, so reconnection
      // is handled there.
    };

    wsRef.current = ws;
  }, [url]);

  const disconnect = useCallback(() => {
    intentionalCloseRef.current = true;
    clearTimer();
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setStatus("disconnected");
  }, [clearTimer]);

  // Auto-connect on mount (if enabled) and clean up on unmount.
  useEffect(() => {
    if (autoConnect) {
      doConnect();
    }
    return () => {
      intentionalCloseRef.current = true;
      clearTimer();
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [autoConnect, doConnect, clearTimer]);

  return { status, connect: doConnect, disconnect };
}
