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
  onSuccess: Exclude<StreamCallback["onSuccess"], undefined>[];
  onError: Exclude<StreamCallback["onError"], undefined>[];
}> | null>(null);

export function useStreamCallback<
  Type extends "onStart" | "onSuccess" | "onError"
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
