import { useState } from "react";
import dayjs from "dayjs";
import ChevronRight from "../assets/ChevronRight.svg?react";
import { RunState } from "../useStreamLog";
import { cn } from "../utils/cn";
import { str } from "../utils/str";

export function IntermediateSteps(props: { latest: RunState }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="flex flex-col border border-divider-700 rounded-2xl bg-background">
      <button
        className="font-medium text-left p-4 flex items-center justify-between"
        onClick={() => setExpanded((open) => !open)}
      >
        <span>Intermediate steps</span>
        <ChevronRight
          className={cn("transition-all", expanded && "rotate-90")}
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
              <pre className="break-words whitespace-pre-wrap min-w-0 text-sm bg-ls-gray-400 rounded-lg p-3">
                {str(log.final_output) ?? "..."}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
