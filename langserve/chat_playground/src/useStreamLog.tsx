import { useCallback, useRef, useState } from "react";
import { applyPatch, Operation } from "fast-json-patch";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { resolveApiUrl } from "./utils/url";
import { StreamCallback } from "./types";

export interface LogEntry {
  // ID of the sub-run.
  id: string;
  // Name of the object being run.
  name: string;
  // Type of the object being run, eg. prompt, chain, llm, etc.
  type: string;
  // List of tags for the run.
  tags: string[];
  // Key-value pairs of metadata for the run.
  metadata: { [key: string]: unknown };
  // ISO-8601 timestamp of when the run started.
  start_time: string;
  // List of LLM tokens streamed by this run, if applicable.
  streamed_output_str: string[];
  // Final output of this run.
  // Only available after the run has finished successfully.
  final_output?: unknown;
  // ISO-8601 timestamp of when the run ended.
  // Only available after the run has finished.
  end_time?: string;
}

export interface RunState {
  // ID of the run.
  id: string;
  // List of output chunks streamed by Runnable.stream()
  streamed_output: unknown[];
  // Final output of the run, usually the result of aggregating (`+`) streamed_output.
  // Only available after the run has finished successfully.
  final_output?: unknown;

  // Map of run names to sub-runs. If filters were supplied, this list will
  // contain only the runs that matched the filters.
  logs: { [name: string]: LogEntry };
}

function reducer(state: RunState | null, action: Operation[]) {
  return applyPatch(state, action, true, false).newDocument;
}

export function useStreamLog(callbacks: StreamCallback = {}) {
  const [latest, setLatest] = useState<RunState | null>(null);
  const [controller, setController] = useState<AbortController | null>(null);

  const startRef = useRef(callbacks.onStart);
  startRef.current = callbacks.onStart;

  const chunkRef = useRef(callbacks.onChunk);
  chunkRef.current = callbacks.onChunk;

  const successRef = useRef(callbacks.onSuccess);
  successRef.current = callbacks.onSuccess;

  const errorRef = useRef(callbacks.onError);
  errorRef.current = callbacks.onError;

  const startStream = useCallback(async (input: unknown, config: unknown) => {
    const controller = new AbortController();
    setController(controller);
    startRef.current?.({ input });

    let innerLatest: RunState | null = null;

    await fetchEventSource(resolveApiUrl("/stream_log").toString(), {
      signal: controller.signal,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input, config }),
      onmessage(msg) {
        if (msg.event === "data") {
          innerLatest = reducer(innerLatest, JSON.parse(msg.data)?.ops);
          setLatest(innerLatest);
          chunkRef.current?.(JSON.parse(msg.data), innerLatest);
        }
      },
      openWhenHidden: true,
      onclose() {
        setController(null);
        successRef.current?.({ input, output: innerLatest?.final_output });
      },
      onerror(error) {
        setController(null);
        errorRef.current?.(error);
        throw error;
      },
    });
  }, []);

  const stopStream = useCallback(() => {
    controller?.abort();
    setController(null);
  }, [controller]);

  return {
    startStream,
    stopStream: controller ? stopStream : undefined,
    latest,
  };
}
