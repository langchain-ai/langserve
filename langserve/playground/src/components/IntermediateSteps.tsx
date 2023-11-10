import { useState } from "react";
import dayjs from "dayjs";
import ChevronRight from "../assets/ChevronRight.svg?react";
import { RunState } from "../useStreamLog";
import { cn } from "../utils/cn";
import { str } from "../utils/str";
import { CorrectnessFeedback } from "./feedback/CorrectnessFeedback";

export function IntermediateSteps(props: {
  latest: RunState;
  feedbackEnabled: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const length = Object.values(props.latest.logs).length;
  const disabled = length === 0;
  return (
    <div className="flex flex-col border border-divider-700 rounded-2xl bg-background">
      <button
        className="font-medium text-left p-4 flex items-center justify-between"
        disabled={disabled}
        onClick={() => setExpanded((open) => !open)}
      >
        <span>
          Intermediate steps{" "}
          <span className="bg-ls-gray-400 text-ls-gray-100 text-sm px-1 py-0.5 rounded-md ml-1">
            {length}
          </span>
        </span>
        <ChevronRight
          className={cn(
            "transition-all",
            expanded && "rotate-90",
            disabled && "opacity-20"
          )}
        />
      </button>
      {expanded && (
        <div className="flex flex-col gap-5 p-4 pt-0 divide-solid divide-y divide-divider-700 rounded-b-xl">
          {Object.values(props.latest.logs).map((log) => (
            <div
              className="gap-3 flex-col min-w-0 flex bg-background pt-3 first-of-type:pt-0"
              key={log.id}
            >
              <div className="flex items-center justify-between">
                <strong className="text-sm font-medium">{log.name}</strong>
                <p className="text-sm">{dayjs.utc(log.start_time).fromNow()}</p>
              </div>
              <div className="bg-ls-gray-400 rounded-lg p-3 relative group">
                <pre className="break-words whitespace-pre-wrap min-w-0 text-sm">
                  {str(log.final_output) ?? "..."}
                </pre>
                {props.feedbackEnabled && log.id ? (
                  <div className="absolute right-3 top-3 flex items-center gap-2 transition-opacity opacity-0 focus-within:opacity-100 group-hover:opacity-100">
                    <CorrectnessFeedback key={log.id} runId={log.id} />
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
