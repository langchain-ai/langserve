import { useState } from "react";
import { CorrectnessFeedback } from "./feedback/CorrectnessFeedback";
import { resolveApiUrl } from "../utils/url";
import { AutosizeTextarea } from "./AutosizeTextarea";
import TrashIcon from "../assets/TrashIcon.svg?react";
import RefreshCW from "../assets/RefreshCW.svg?react";

export type ChatMessageType = "human" | "ai" | "function" | "tool" | "system";

export type ChatMessageBody = {
  type: ChatMessageType;
  content: string;
  runId?: string;
}

export function ChatMessage(props: {
  message: ChatMessageBody;
  isLoading?: boolean;
  onError?: (e: any) => void;
  onTypeChange?: (newValue: string) => void;
  onChange?: (newValue: string) => void;
  onRemove?: (e: any) => void;
  onRegenerate?: (e?: any) => void;
  isFinalMessage?: boolean;
  feedbackEnabled?: boolean;
  publicTraceLinksEnabled?: boolean;
}) {
  const { message, feedbackEnabled, publicTraceLinksEnabled, onError, isLoading } = props;
  const { content, type, runId } = message;

  const [publicTraceLink, setPublicTraceLink] = useState<string | null>(null);
  const [messageActionIsLoading, setMessageActionIsLoading] = useState(false);
  const openPublicTrace = async () => {
    if (messageActionIsLoading) {
      return;
    }
    if (publicTraceLink) {
      window.open(publicTraceLink, '_blank');
      return;
    }
    setMessageActionIsLoading(true);
    const payload = { run_id: runId };
    const response = await fetch(resolveApiUrl("/public_trace_link"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      if (response.status === 404) {
        onError?.(new Error(`Feedback endpoint not found. Please enable it in your LangServe endpoint.`));
      } else {
        try {
          const errorResponse = await response.json();
          onError?.(new Error(`${errorResponse.detail}`));
        } catch (e) {
          onError?.(new Error(`Request failed with status: ${response.status}`));
        }
      }
      setMessageActionIsLoading(false);
      throw new Error(`Failed request ${response.status}`)
    }
    const parsedResponse = await response.json();
    setMessageActionIsLoading(false);
    setPublicTraceLink(parsedResponse.public_url);
    window.open(parsedResponse.public_url, '_blank');
  };

  return (
    <div className="mb-8 group">
      <div className="flex justify-between">
        <select
          className="font-medium text-transform uppercase mb-2 appearance-none"
          defaultValue={type}
          onChange={(e) => props.onTypeChange?.(e.target.value)}
        >
          <option value="human">HUMAN</option>
          <option value="ai">AI</option>
          <option value="system">SYSTEM</option>
        </select>
        <span className="flex">
          {props.isFinalMessage &&
            type === "human" && 
            <RefreshCW className="opacity-0 group-hover:opacity-50 transition-opacity duration-200 cursor-pointer h-4 w-4 mr-2" onMouseUp={props.onRegenerate}></RefreshCW>}
          <TrashIcon
            className="opacity-0 group-hover:opacity-50 transition-opacity duration-200 cursor-pointer h-4 w-4"
            onMouseUp={props.onRemove}
          ></TrashIcon>
        </span>
      </div>
      <AutosizeTextarea value={content} fullHeight={true} onChange={props.onChange} onKeyDown={(e) => {
        if (
          e.key === 'Enter' &&
          !e.shiftKey &&
          props.isFinalMessage &&
          type === "human"
        ) {
          e.preventDefault();
          props.onRegenerate?.();
        }
      }}></AutosizeTextarea>
      {type === "ai" && !isLoading && runId != null && (
        <div className="mt-2 flex items-center">
          {feedbackEnabled && <span className="mr-2"><CorrectnessFeedback runId={runId} onError={props.onError}></CorrectnessFeedback></span>}
          {publicTraceLinksEnabled && <>
            <button
              className="bg-button-inline p-2 rounded-lg text-xs font-medium hover:opacity-80"
              disabled={messageActionIsLoading || isLoading}
              onMouseUp={openPublicTrace}
            >
              🛠️ View LangSmith trace
            </button>
          </>}
        </div>
      )}
    </div> 
  )
};
