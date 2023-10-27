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

    return (
      <div className="control">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase font-semibold text-ls-gray-100">
            {props.label || "Messages"}
          </label>
          <button
            className="p-1 rounded-full"
            onClick={() => {
              const lastRole = data.length ? data[data.length - 1][0] : "ai";
              props.handleChange(props.path, [
                ...data,
                [lastRole === "ai" ? "human" : "ai", ""],
              ]);
            }}
          >
            <PlusIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="flex flex-col gap-3 mt-1 empty:hidden">
          {data.map(([type, content], index) => {
            const msgPath = Paths.compose(props.path, `${index}`);
            return (
              <div className="control group" key={index}>
                <div className="flex items-start justify-between gap-2">
                  <select
                    className="-ml-1 min-w-[100px]"
                    value={type}
                    onChange={(e) => {
                      props.handleChange(
                        Paths.compose(msgPath, "0"),
                        e.target.value
                      );
                    }}
                  >
                    <option value="human">Human</option>
                    <option value="ai">AI</option>
                    <option value="system">System</option>
                  </select>
                  <button
                    className="p-1 border rounded opacity-0 transition-opacity border-divider-700 group-focus-within:opacity-100 group-hover:opacity-100"
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

                <AutosizeTextarea
                  value={content}
                  onChange={(content) => {
                    props.handleChange(Paths.compose(msgPath, "1"), content);
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>
    );
  }
);
