import {
  MutableRefObject,
  createContext,
  useContext,
  useEffect,
  useRef,
} from "react";

import { StreamCallback } from "./types";

export const AppCallbackContext = createContext<MutableRefObject<{
  onStart: Exclude<StreamCallback["onStart"], undefined>[];
  onChunk: Exclude<StreamCallback["onChunk"], undefined>[];
  onSuccess: Exclude<StreamCallback["onSuccess"], undefined>[];
  onError: Exclude<StreamCallback["onError"], undefined>[];
}> | null>(null);

export function useAppStreamCallbacks() {
  // callbacks handling
  const context = useRef<{
    onStart: Exclude<StreamCallback["onStart"], undefined>[];
    onChunk: Exclude<StreamCallback["onChunk"], undefined>[];
    onSuccess: Exclude<StreamCallback["onSuccess"], undefined>[];
    onError: Exclude<StreamCallback["onError"], undefined>[];
  }>({ onStart: [], onChunk: [], onSuccess: [], onError: [] });

  const callbacks: StreamCallback = {
    onStart(...args) {
      for (const callback of context.current.onStart) {
        callback(...args);
      }
    },
    onChunk(...args) {
      for (const callback of context.current.onChunk) {
        callback(...args);
      }
    },
    onSuccess(...args) {
      for (const callback of context.current.onSuccess) {
        callback(...args);
      }
    },
    onError(...args) {
      for (const callback of context.current.onError) {
        callback(...args);
      }
    },
  };

  return { context, callbacks };
}

export function useStreamCallback<
  Type extends "onStart" | "onChunk" | "onSuccess" | "onError"
>(type: Type, callback: Exclude<StreamCallback[Type], undefined>) {
  type CallbackType = Exclude<StreamCallback[Type], undefined>;

  const appCbRef = useContext(AppCallbackContext);

  const callbackRef = useRef<CallbackType>(callback);
  callbackRef.current = callback;

  useEffect(() => {
    // @ts-expect-error Not sure why I can't expand the tuple
    const current = (...args) => callbackRef.current?.(...args);
    appCbRef?.current[type].push(current);

    return () => {
      if (!appCbRef) return;

      // @ts-expect-error Assingability issues due to the tuple object
      // eslint-disable-next-line react-hooks/exhaustive-deps
      appCbRef.current[type] = appCbRef.current[type].filter(
        (callbacks) => callbacks !== current
      );
    };
  }, [type, appCbRef]);
}
