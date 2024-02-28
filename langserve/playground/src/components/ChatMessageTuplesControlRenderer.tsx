import { withJsonFormsControlProps } from "@jsonforms/react";
import PlusIcon from "../assets/PlusIcon.svg?react";
import TrashIcon from "../assets/TrashIcon.svg?react";
import {
  rankWith,
  and,
  schemaMatches,
  Paths,
  isControl,
} from "@jsonforms/core";
import { AutosizeTextarea } from "./AutosizeTextarea";
import { isJsonSchemaExtra } from "../utils/schema";
import { useStreamCallback } from "../useStreamCallback";
import { getNormalizedJsonPath, traverseNaiveJsonPath } from "../utils/path";
import { getMessageContent } from "../utils/messages";

type MessageTuple = [string, string];

export const chatMessagesTupleTester = rankWith(
  12,
  and(
    isControl,
    schemaMatches((schema) => {
      if (schema.type !== "array") return false;
      if (typeof schema.items !== "object" || schema.items == null)
        return false;

      if (!isJsonSchemaExtra(schema) || schema.extra.widget.type !== "chat") {
        return false;
      }

      if ("type" in schema.items) {
        return (
          schema.items.type === "array" &&
          schema.items.minItems === 2 &&
          schema.items.maxItems === 2 &&
          Array.isArray(schema.items.items) &&
          schema.items.items.length === 2 &&
          schema.items.items.every((schema) => schema.type === "string")
        );
      }

      return false;
    })
  )
);

export const ChatMessageTuplesControlRenderer = withJsonFormsControlProps(
  (props) => {
    const data: Array<MessageTuple> = props.data ?? [];

    useStreamCallback("onChunk", (_chunk, aggregatedState) => {
      if (!isJsonSchemaExtra(props.schema)) return;
      const widget = props.schema.extra.widget;
      if (!("input" in widget) && !("output" in widget)) return;

      const outputPath = getNormalizedJsonPath(widget.output ?? "");

      const isSingleOutputKey =
        aggregatedState?.final_output != null &&
        Object.keys(aggregatedState?.final_output).length === 1 &&
        Object.keys(aggregatedState?.final_output)[0] === "output";

      let ai = traverseNaiveJsonPath(aggregatedState?.final_output, outputPath);

      if (isSingleOutputKey) {
        ai = traverseNaiveJsonPath(ai, ["output", ...outputPath]) ?? ai;
      }

      ai = getMessageContent(ai);
      if (typeof ai === "string") {
        props.handleChange(props.path, [...data.slice(0, -1), [data[data.length - 1][0], ai]]);
      }
    });

    useStreamCallback("onStart", (ctx) => {
      if (!isJsonSchemaExtra(props.schema)) return;
      const widget = props.schema.extra.widget;
      if (!("input" in widget) && !("output" in widget)) return;

      const inputPath = getNormalizedJsonPath(widget.input ?? "");

      const human = traverseNaiveJsonPath(ctx.input, inputPath);
      if (typeof human === "string") {
        props.handleChange(props.path, [...data, [human, ""]]);
      }
    });

    useStreamCallback("onError", () => {
      props.handleChange(props.path, [...data.slice(0, -1)]);
    });

    return (
      <div className="control">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase font-semibold text-ls-gray-100">
            {props.label || "Messages"}
          </label>
          <button
            className="p-1 rounded-full"
            onClick={() => props.handleChange(props.path, [...data, ["", ""]])}
          >
            <PlusIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="flex flex-col gap-3 mt-1 empty:hidden">
          {data.map(([human, ai], index) => {
            const msgPath = Paths.compose(props.path, `${index}`);
            return (
              <div className="control group relative" key={index}>
                <div className="grid gap-3">
                  <div className="flex-grow">
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-xs uppercase font-semibold text-ls-gray-100 mb-1 ">
                        Human
                      </div>
                    </div>
                    <AutosizeTextarea
                      value={human}
                      onChange={(human) => {
                        props.handleChange(Paths.compose(msgPath, "0"), human);
                      }}
                    />
                  </div>
                  <div className="flex-shrink-0 h-px bg-divider-700" />
                  <div className="flex-grow">
                    <div className="flex items-start justify-between gap-2">
                      <div className="text-xs uppercase font-semibold text-ls-gray-100 mb-1 ">
                        AI
                      </div>
                    </div>

                    <AutosizeTextarea
                      value={ai}
                      onChange={(ai) => {
                        props.handleChange(Paths.compose(msgPath, "1"), ai);
                      }}
                    />
                  </div>
                </div>

                <button
                  className="absolute right-3 top-3 p-1 border rounded opacity-0 transition-opacity border-divider-700 group-focus-within:opacity-100 group-hover:opacity-100"
                  onClick={() => {
                    props.handleChange(
                      props.path,
                      data.filter((_, i) => i !== index)
                    );
                  }}
                >
                  <TrashIcon className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      </div>
    );
  }
);
