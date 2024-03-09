import { useState } from "react";
import { CorrectnessFeedback } from "./feedback/CorrectnessFeedback";
import { resolveApiUrl } from "../utils/url";

export type ChatMessageType = {
  role: "human" | "ai" | "function" | "tool" | "system";
  content: string;
  runId?: string;
}

export function ChatMessage(props: {
  message: ChatMessageType;
  isLoading?: boolean;
  onError?: (e: any) => void;
  feedbackEnabled?: boolean;
  publicTraceLinksEnabled?: boolean;
}) {
  const { message, feedbackEnabled, publicTraceLinksEnabled, onError, isLoading } = props;
  const { content, role, runId } = message;

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
    <div className="mb-8">
      <p className="font-medium text-transform uppercase mb-2">{role}</p>
      <p>{content}</p>
      {role === "ai" && !isLoading && runId != null && (
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
