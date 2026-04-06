import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";
import type {
  PipelineState,
  DemoHighlight,
  TrainingMetrics,
  ModelSummary,
  DatasetStats,
  EvalResult,
  ComparisonResult,
  GenerationResult,
  WSMessage,
} from "../types";
import { useWebSocket, type ConnectionStatus } from "../hooks/useWebSocket";

// ---------------------------------------------------------------------------
// Application state
// ---------------------------------------------------------------------------

export interface AppState {
  pipeline: PipelineState;
  modelSummary: ModelSummary | null;
  datasetStats: DatasetStats | null;
  evalResult: EvalResult | null;
  comparisonResult: ComparisonResult | null;
  generationResult: GenerationResult | null;
  metricsHistory: TrainingMetrics[];
  commentary: string[];
  error: string | null;
  progress: string | null;
  connectionStatus: ConnectionStatus;
}

const initialPipeline: PipelineState = {
  stage: "idle",
  isDemo: false,
  demoSpeed: "medium",
  demoPaused: false,
  demoHighlight: null,
};

export const initialState: AppState = {
  pipeline: initialPipeline,
  modelSummary: null,
  datasetStats: null,
  evalResult: null,
  comparisonResult: null,
  generationResult: null,
  metricsHistory: [],
  commentary: [],
  error: null,
  progress: null,
  connectionStatus: "disconnected",
};

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export type AppAction =
  | { type: "SET_CONNECTION_STATUS"; status: ConnectionStatus }
  | { type: "SET_PIPELINE_STAGE"; stage: PipelineState["stage"] }
  | { type: "SET_DEMO_STATE"; isDemo: boolean; demoSpeed?: PipelineState["demoSpeed"]; demoPaused?: boolean; demoHighlight?: DemoHighlight }
  | { type: "SET_MODEL_SUMMARY"; summary: ModelSummary }
  | { type: "SET_DATASET_STATS"; stats: DatasetStats }
  | { type: "SET_EVAL_RESULT"; result: EvalResult }
  | { type: "SET_COMPARISON_RESULT"; result: ComparisonResult }
  | { type: "SET_GENERATION_RESULT"; result: GenerationResult }
  | { type: "ADD_METRICS"; metrics: TrainingMetrics }
  | { type: "CLEAR_METRICS" }
  | { type: "ADD_COMMENTARY"; text: string }
  | { type: "SET_ERROR"; message: string | null }
  | { type: "SET_PROGRESS"; message: string | null }
  | { type: "WS_MESSAGE"; message: WSMessage };

// ---------------------------------------------------------------------------
// Reducer
// ---------------------------------------------------------------------------

export function appReducer(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "SET_CONNECTION_STATUS":
      return { ...state, connectionStatus: action.status };

    case "SET_PIPELINE_STAGE":
      return {
        ...state,
        pipeline: { ...state.pipeline, stage: action.stage },
        error: null,
      };

    case "SET_DEMO_STATE":
      return {
        ...state,
        pipeline: {
          ...state.pipeline,
          isDemo: action.isDemo,
          demoSpeed: action.demoSpeed ?? state.pipeline.demoSpeed,
          demoPaused: action.demoPaused ?? state.pipeline.demoPaused,
          demoHighlight: action.demoHighlight ?? (action.isDemo ? state.pipeline.demoHighlight : null),
        },
      };

    case "SET_MODEL_SUMMARY":
      return { ...state, modelSummary: action.summary };

    case "SET_DATASET_STATS":
      return { ...state, datasetStats: action.stats };

    case "SET_EVAL_RESULT":
      return { ...state, evalResult: action.result };

    case "SET_COMPARISON_RESULT":
      return { ...state, comparisonResult: action.result };

    case "SET_GENERATION_RESULT":
      return { ...state, generationResult: action.result };

    case "ADD_METRICS":
      return {
        ...state,
        metricsHistory: [...state.metricsHistory, action.metrics],
      };

    case "CLEAR_METRICS":
      return { ...state, metricsHistory: [] };

    case "ADD_COMMENTARY":
      return { ...state, commentary: [...state.commentary, action.text] };

    case "SET_ERROR":
      return { ...state, error: action.message };

    case "SET_PROGRESS":
      return { ...state, progress: action.message };

    case "WS_MESSAGE":
      return handleWSMessage(state, action.message);

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// WebSocket message → state mapping
// ---------------------------------------------------------------------------

function handleWSMessage(state: AppState, msg: WSMessage): AppState {
  switch (msg.type) {
    case "state_change": {
      const stage = msg.payload.stage as PipelineState["stage"] | undefined;
      if (stage) {
        return {
          ...state,
          pipeline: { ...state.pipeline, stage },
          error: null,
        };
      }
      return state;
    }

    case "metrics": {
      const m = msg.payload as unknown as TrainingMetrics;
      return {
        ...state,
        metricsHistory: [...state.metricsHistory, m],
      };
    }

    case "progress": {
      const message = (msg.payload.message as string) ?? null;
      return { ...state, progress: message };
    }

    case "commentary": {
      const text = msg.payload.text as string;
      if (text) {
        return { ...state, commentary: [...state.commentary, text] };
      }
      return state;
    }

    case "error": {
      const errorMsg =
        (msg.payload.message as string) ??
        (msg.payload.error_code as string) ??
        "Unknown error";
      return { ...state, error: errorMsg };
    }

    case "demo_step": {
      const stage = msg.payload.stage as PipelineState["stage"] | undefined;
      const paused = msg.payload.paused as boolean | undefined;
      const highlight = (msg.payload.highlight as DemoHighlight) ?? null;
      return {
        ...state,
        pipeline: {
          ...state.pipeline,
          ...(stage ? { stage } : {}),
          ...(paused !== undefined ? { demoPaused: paused } : {}),
          demoHighlight: highlight,
        },
      };
    }

    default:
      return state;
  }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

interface AppContextValue {
  state: AppState;
  dispatch: React.Dispatch<AppAction>;
  wsStatus: ConnectionStatus;
  wsConnect: () => void;
  wsDisconnect: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

const WS_URL = "ws://localhost:8000/ws";

export function AppProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  const onMessage = useCallback(
    (msg: WSMessage) => {
      dispatch({ type: "WS_MESSAGE", message: msg });
    },
    [],
  );

  const { status: wsStatus, connect: wsConnect, disconnect: wsDisconnect } =
    useWebSocket({ url: WS_URL, onMessage });

  // Sync connection status into state so components can read it from one place.
  useEffect(() => {
    dispatch({ type: "SET_CONNECTION_STATUS", status: wsStatus });
  }, [wsStatus]);

  return (
    <AppContext.Provider
      value={{ state, dispatch, wsStatus, wsConnect, wsDisconnect }}
    >
      {children}
    </AppContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

export function useAppState(): AppContextValue {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error("useAppState must be used within an <AppProvider>");
  }
  return ctx;
}
