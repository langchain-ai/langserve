import { useState } from "react";
import { CorrectnessFeedback } from "./feedback/CorrectnessFeedback";
import { resolveApiUrl } from "../utils/url";

export type ChatMessageType = {
  role: "human" | "ai" | "function" | "tool" | "system";
  content: string;
  runId?: string;
}

export function ChatMessage(props: { message: ChatMessageType, isLoading?: boolean, onError?: (e: any) => void }) {
  const { content, role, runId } = props.message;

  const [publicTraceLink, setPublicTraceLink] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const openPublicTrace = async () => {
    if (isLoading) {
      return;
    }
    if (publicTraceLink) {
      window.open(publicTraceLink, '_blank');
      return;
    }
    setIsLoading(true);
    const payload = { run_id: runId };
    const response = await fetch(resolveApiUrl("/public_trace_link"), {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      if (response.status === 404) {
        props.onError?.(new Error(`Feedback endpoint not found. Please enable it in your LangServe endpoint.`));
      } else {
        try {
          const errorResponse = await response.json();
          props.onError?.(new Error(`${errorResponse.detail}`));
        } catch (e) {
          props.onError?.(new Error(`Request failed with status: ${response.status}`));
        }
      }
      setIsLoading(false);
      throw new Error(`Failed request ${response.status}`)
    }
    const parsedResponse = await response.json();
    setIsLoading(false);
    setPublicTraceLink(parsedResponse.public_url);
    window.open(parsedResponse.public_url, '_blank');
  };

  return (
    <div className="mb-8">
      <p className="font-medium text-transform uppercase mb-2">{role}</p>
      <p>{content}</p>
      {role === "ai" && !props.isLoading && runId != null && (
        <div className="mt-2 flex">
          <CorrectnessFeedback runId={runId} onError={props.onError}></CorrectnessFeedback>
          <>
            <button
              className="ml-2 bg-button-inline p-2 rounded-lg text-xs font-medium hover:opacity-80"
              disabled={isLoading}
              onMouseUp={openPublicTrace}
            >
              🛠️ View LangSmith trace
            </button>
          </>
        </div>
      )}
    </div> 
  )
};
