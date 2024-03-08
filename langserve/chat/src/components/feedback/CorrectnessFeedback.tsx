import ThumbsUpIcon from "../../assets/ThumbsUpIcon.svg?react";
import ThumbsDownIcon from "../../assets/ThumbsDownIcon.svg?react";
import CircleSpinIcon from "../../assets/CircleSpinIcon.svg?react";
import { resolveApiUrl } from "../../utils/url";
import { useState } from "react";
import { cn } from "../../utils/cn";
import useSWRMutation from "swr/mutation";

const useFeedbackMutation = (runId: string) => {
  interface FeedbackArguments {
    key: string;
    score: number;
  }

  const [lastArg, setLastArg] = useState<FeedbackArguments | null>(null);

  const mutation = useSWRMutation(
    ["feedback", runId],
    async ([, runId], { arg }: { arg: FeedbackArguments }) => {
      const payload = { run_id: runId, key: arg.key, score: arg.score };
      setLastArg(arg);

      const request = await fetch(resolveApiUrl("/feedback"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!request.ok) throw new Error(`Failed request ${request.status}`);
      const json: {
        id: string;
        score: number;
      } = await request.json();

      return json;
    }
  );

  return { lastArg: mutation.isMutating ? lastArg : null, mutation };
};

export function CorrectnessFeedback(props: { runId: string }) {
  const score = useFeedbackMutation(props.runId);

  if (props.runId == null) return null;
  return (
    <>
      <button
        type="button"
        className={cn(
          "border focus-within:border-ls-blue focus-within:outline-none bg-background rounded p-1 border-divider-700 hover:bg-divider-500/50 active:bg-divider-500",
          score.mutation.data?.score === 1 && "text-teal-500"
        )}
        disabled={score.mutation.isMutating}
        onClick={() => {
          if (score.mutation.data?.score !== 1) {
            score.mutation.trigger({ key: "correctness", score: 1 });
          }
        }}
      >
        {score.lastArg?.score === 1 ? (
          <CircleSpinIcon className="animate-spin w-4 h-4 text-white/50 fill-white" />
        ) : (
          <ThumbsUpIcon className="w-4 h-4" />
        )}
      </button>

      <button
        type="button"
        className={cn(
          "border focus-within:border-ls-blue focus-within:outline-none bg-background rounded p-1 border-divider-700 hover:bg-divider-500/50 active:bg-divider-500",
          score.mutation.data?.score === 0 && "text-red-500"
        )}
        disabled={score.mutation.isMutating}
        onClick={() => {
          if (score.mutation.data?.score !== 0) {
            score.mutation.trigger({ key: "correctness", score: 0 });
          }
        }}
      >
        {score.lastArg?.score === 0 ? (
          <CircleSpinIcon className="animate-spin w-4 h-4 text-white/50 fill-white" />
        ) : (
          <ThumbsDownIcon className="w-4 h-4" />
        )}
      </button>
    </>
  );
}
