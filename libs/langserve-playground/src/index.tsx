"use client";
import React, { useEffect } from "react";
import { useLayoutEffect, useRef, useState } from "react";

export function LangServePlayground(props: {
  baseUrl: string;
  value: Record<string, unknown>;
  onChange: (value: Record<string, unknown>) => void;
}) {
  const [open, setOpen] = useState(false);

  const ref = useRef<HTMLIFrameElement>(null);

  const onChangeRef = useRef(props.onChange);
  onChangeRef.current = props.onChange;

  const onUpdateRef = useRef<(() => void) | null>(null);
  onUpdateRef.current = () =>
    ref.current?.contentWindow?.postMessage(
      { type: "update", value: { config: props.value } },
      "*"
    );

  useEffect(() => void onUpdateRef.current?.(), [props.value, open]);
  useLayoutEffect(() => {
    function listener(event: MessageEvent) {
      // Check the event origin to ensure it comes from the expected iframe
      if (event.source === ref.current?.contentWindow) {
        const message = event.data;

        if (typeof message === "object" && message != null) {
          switch (message.type) {
            case "init": {
              onUpdateRef.current?.();
              break;
            }
            case "close": {
              setOpen(false);
              break;
            }
            case "apply": {
              const value: {
                targetUrl: string;
                config: Record<string, unknown>;
              } = message.value;

              onChangeRef.current?.(value.config);
              setOpen(false);
              break;
            }
          }
        }
      }
    }

    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, []);

  let iframeSrc = props.baseUrl;
  if (iframeSrc.endsWith("/")) iframeSrc = iframeSrc.slice(0, -1);
  iframeSrc += "/playground";

  return (
    <div
      style={{
        display: "flex",
        position: "fixed",
        bottom: "1rem",
        right: "1rem",
        flexDirection: "column",
        alignItems: "flex-end",
        gap: "1rem",
      }}
    >
      {open && (
        <iframe
          src={iframeSrc}
          style={{
            minWidth: 360,
            minHeight: 600,
            borderRadius: 16,
            background: "#fff",
          }}
          ref={ref}
        />
      )}
      <button
        style={{
          border: "none",
          background: "#fff",
          width: 52,
          height: 52,
          fontSize: 24,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRadius: "100%",
          cursor: "pointer",
        }}
        onClick={() => {
          setOpen((open) => !open);
        }}
      >
        🦜
      </button>
    </div>
  );
}
