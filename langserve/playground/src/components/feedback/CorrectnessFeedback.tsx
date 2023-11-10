import ThumbsUpIcon from "../../assets/ThumbsUpIcon.svg?react";
import ThumbsDownIcon from "../../assets/ThumbsDownIcon.svg?react";
import CircleSpinIcon from "../../assets/CircleSpinIcon.svg?react";
import { resolveApiUrl } from "../../utils/url";
import { useEffect, useRef, useState } from "react";
import { cn } from "../../utils/cn";

export function CorrectnessFeedback(props: { runId?: string }) {
  const [loading, setLoading] = useState<number | null>(null);
  const [state, setState] = useState<number | null>(null);

  const isMounted = useRef<boolean>(true);
  useEffect(() => {
    isMounted.current = true;
    return () => {
      isMounted.current = false;
    };
  }, []);

  const sendFeedback = async (payload: {
    run_id: string;
    key: string;
    score: number;
  }) => {
    if (isMounted.current) setLoading(payload.score);

    try {
      const request = await fetch(resolveApiUrl("/feedback"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!request.ok) throw new Error(`Failed request ${request.status}`);
      const json: {
        score: number;
      } = await request.json();

      if (isMounted.current) setState(json.score);
    } finally {
      if (isMounted.current) setLoading(null);
    }
  };

  if (props.runId == null) return null;
  return (
    <>
      <button
        type="button"
        className={cn(
          "border focus-within:border-ls-blue focus-within:outline-none bg-background rounded p-1 border-divider-700 hover:bg-divider-500/50 active:bg-divider-500",
          state === 1 && "text-teal-500"
        )}
        disabled={loading != null}
        onClick={() => {
          if (props.runId) {
            sendFeedback({
              run_id: props.runId,
              key: "correctness",
              score: 1,
            });
          }
        }}
      >
        {loading === 1 ? (
          <CircleSpinIcon className="animate-spin w-4 h-4 text-white/50 fill-white" />
        ) : (
          <ThumbsUpIcon className="w-4 h-4" />
        )}
      </button>

      <button
        type="button"
        className={cn(
          "border focus-within:border-ls-blue focus-within:outline-none bg-background rounded p-1 border-divider-700 hover:bg-divider-500/50 active:bg-divider-500",
          state === -1 && "text-red-500"
        )}
        disabled={loading != null}
        onClick={() => {
          if (props.runId) {
            sendFeedback({
              run_id: props.runId,
              key: "correctness",
              score: -1,
            });
          }
        }}
      >
        {loading === -1 ? (
          <CircleSpinIcon className="animate-spin w-4 h-4 text-white/50 fill-white" />
        ) : (
          <ThumbsDownIcon className="w-4 h-4" />
        )}
      </button>
    </>
  );
}
