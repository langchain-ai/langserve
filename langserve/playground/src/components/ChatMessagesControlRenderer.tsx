import { withJsonFormsControlProps } from "@jsonforms/react";
import PlusIcon from "../assets/PlusIcon.svg?react";
import TrashIcon from "../assets/TrashIcon.svg?react";
import CodeIcon from "../assets/CodeIcon.svg?react";
import ChatIcon from "../assets/ChatIcon.svg?react";
import {
  rankWith,
  and,
  schemaMatches,
  Paths,
  isControl,
} from "@jsonforms/core";
import { AutosizeTextarea } from "./AutosizeTextarea";
import { useStreamCallback } from "../useStreamCallback";
import { getNormalizedJsonPath, traverseNaiveJsonPath } from "../utils/path";
import { isJsonSchemaExtra } from "../utils/schema";
import * as ToggleGroup from "@radix-ui/react-toggle-group";

export const chatMessagesTester = rankWith(
  12,
  and(
    isControl,
    schemaMatches((schema) => {
      if (schema.type !== "array") return false;
      if (typeof schema.items !== "object" || schema.items == null)
        return false;

      if (
        "type" in schema.items &&
        schema.items.type != null &&
        schema.items.title != null
      ) {
        return (
          schema.items.type === "object" &&
          (schema.items.title?.endsWith("Message") ||
            schema.items.title?.endsWith("MessageChunk"))
        );
      }

      if ("anyOf" in schema.items && schema.items.anyOf != null) {
        return schema.items.anyOf.every((schema) => {
          const isObjectMessage =
            schema.type === "object" &&
            (schema.title?.endsWith("Message") ||
              schema.title?.endsWith("MessageChunk"));

          const isTupleMessage =
            schema.type === "array" &&
            schema.minItems === 2 &&
            schema.maxItems === 2 &&
            Array.isArray(schema.items) &&
            schema.items.length === 2 &&
            schema.items.every((schema) => schema.type === "string");

          return isObjectMessage || isTupleMessage;
        });
      }

      return false;
    })
  )
);

interface MessageFields {
  content: string;
  additional_kwargs?: { [key: string]: unknown };
  name?: string;
  type?: string;

  role?: string;
}

function isMessageFields(x: unknown): x is MessageFields {
  if (typeof x !== "object" || x == null) return false;
  if (!("content" in x) || typeof x.content !== "string") return false;
  if (
    "additional_kwargs" in x &&
    typeof x.additional_kwargs !== "object" &&
    x.additional_kwargs != null
  )
    return false;
  if ("name" in x && typeof x.name !== "string" && x.name != null) return false;
  if ("type" in x && typeof x.type !== "string" && x.type != null) return false;
  if ("role" in x && typeof x.role !== "string" && x.role != null) return false;
  return true;
}

function constructMessage(
  x: unknown,
  assumedRole: string
): Array<MessageFields> | null {
  if (typeof x === "string") {
    return [{ content: x, type: assumedRole }];
  }

  if (isMessageFields(x)) {
    return [x];
  }

  if (Array.isArray(x) && x.every(isMessageFields)) {
    return x;
  }

  return null;
}

function isOpenAiFunctionCall(
  x: unknown
): x is { name: string; arguments: string } {
  if (typeof x !== "object" || x == null) return false;
  if (!("name" in x) || typeof x.name !== "string") return false;
  if (!("arguments" in x) || typeof x.arguments !== "string") return false;
  return true;
}

export const ChatMessagesControlRenderer = withJsonFormsControlProps(
  (props) => {
    const data: Array<MessageFields> = props.data ?? [];

    useStreamCallback("onSuccess", (ctx) => {
      if (!isJsonSchemaExtra(props.schema)) return;
      const widget = props.schema.extra.widget;
      if (!("input" in widget) && !("output" in widget)) return;

      const inputPath = getNormalizedJsonPath(widget.input ?? "");
      const outputPath = getNormalizedJsonPath(widget.output ?? "");

      const human = traverseNaiveJsonPath(ctx.input, inputPath);
      let ai = traverseNaiveJsonPath(ctx.output, outputPath);

      const isSingleOutputKey =
        ctx.output != null &&
        Object.keys(ctx.output).length === 1 &&
        Object.keys(ctx.output)[0] === "output";

      if (isSingleOutputKey) {
        ai = traverseNaiveJsonPath(ai, ["output", ...outputPath]) ?? ai;
      }

      const humanMsg = constructMessage(human, "human");
      const aiMsg = constructMessage(ai, "ai");

      let newMessages = undefined;
      if (humanMsg != null) {
        newMessages ??= [...data];
        newMessages.push(...humanMsg);
      }
      if (aiMsg != null) {
        newMessages ??= [...data];
        newMessages.push(...aiMsg);
      }

      if (newMessages != null) {
        props.handleChange(props.path, newMessages);
      }
    });

    return (
      <div className="control">
        <div className="flex items-center justify-between">
          <label className="text-xs uppercase font-semibold text-ls-gray-100">
            {props.label || "Messages"}
          </label>
          <button
            className="p-1 rounded-full"
            onClick={() => {
              const lastRole = data.length ? data[data.length - 1].type : "ai";
              props.handleChange(props.path, [
                ...data,
                { content: "", type: lastRole === "human" ? "ai" : "human" },
              ]);
            }}
          >
            <PlusIcon className="w-5 h-5" />
          </button>
        </div>

        <div className="flex flex-col gap-3 mt-1 empty:hidden">
          {data.map((message, index) => {
            const msgPath = Paths.compose(props.path, `${index}`);
            const type = message.type ?? "chat";

            const isAiFunctionCall = isOpenAiFunctionCall(
              message.additional_kwargs?.function_call
            );

            return (
              <div className="control group" key={index}>
                <div className="flex items-start justify-between gap-2">
                  <select
                    className="-ml-1 min-w-[100px]"
                    value={type}
                    onChange={(e) => {
                      props.handleChange(
                        Paths.compose(msgPath, "type"),
                        e.target.value
                      );
                    }}
                  >
                    <option value="human">Human</option>
                    <option value="ai">AI</option>
                    <option value="system">System</option>
                    <option value="function">Function</option>

                    <option value="chat">Chat</option>
                  </select>
                  <div className="flex items-center gap-2">
                    {message.type === "ai" && (
                      <ToggleGroup.Root
                        type="single"
                        aria-label="Message Type"
                        className="opacity-0 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100"
                        value={isAiFunctionCall ? "function" : "text"}
                        onValueChange={(value) => {
                          switch (value) {
                            case "function": {
                              props.handleChange(
                                Paths.compose(msgPath, "additional_kwargs"),
                                {
                                  function_call: {
                                    name: "",
                                    arguments: "{}",
                                  },
                                }
                              );

                              break;
                            }
                            case "text": {
                              props.handleChange(
                                Paths.compose(msgPath, "additional_kwargs"),
                                {}
                              );

                              break;
                            }
                          }
                        }}
                      >
                        <ToggleGroup.Item
                          className="rounded-s border border-divider-700 px-2.5 py-1 data-[state=on]:bg-divider-500/50"
                          value="text"
                          aria-label="Text message"
                        >
                          <ChatIcon className="w-4 h-4" />
                        </ToggleGroup.Item>
                        <ToggleGroup.Item
                          className="rounded-e border border-l-0 border-divider-700 px-2.5 py-1 data-[state=on]:bg-divider-500/50"
                          value="function"
                          aria-label="Function call"
                        >
                          <CodeIcon className="w-4 h-4" />
                        </ToggleGroup.Item>
                      </ToggleGroup.Root>
                    )}

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
                </div>

                {type === "chat" && (
                  <input
                    className="mb-1"
                    placeholder="Role"
                    value={message.role ?? ""}
                    onChange={(e) => {
                      props.handleChange(
                        Paths.compose(msgPath, "role"),
                        e.target.value
                      );
                    }}
                  />
                )}

                {type === "function" && (
                  <input
                    className="mb-1"
                    placeholder="Function Name"
                    value={message.name ?? ""}
                    onChange={(e) => {
                      props.handleChange(
                        Paths.compose(msgPath, "name"),
                        e.target.value
                      );
                    }}
                  />
                )}

                {type === "ai" &&
                isOpenAiFunctionCall(
                  message.additional_kwargs?.function_call
                ) ? (
                  <>
                    <input
                      className="mb-1"
                      placeholder="Function Name"
                      value={
                        message.additional_kwargs?.function_call.name ?? ""
                      }
                      onChange={(e) => {
                        props.handleChange(
                          Paths.compose(
                            msgPath,
                            "additional_kwargs.function_call.name"
                          ),
                          e.target.value
                        );
                      }}
                    />

                    <AutosizeTextarea
                      value={
                        message.additional_kwargs?.function_call?.arguments ??
                        ""
                      }
                      onChange={(content) => {
                        props.handleChange(
                          Paths.compose(
                            msgPath,
                            "additional_kwargs.function_call.arguments"
                          ),
                          content
                        );
                      }}
                    />
                  </>
                ) : (
                  <AutosizeTextarea
                    value={message.content}
                    onChange={(content) => {
                      props.handleChange(
                        Paths.compose(msgPath, "content"),
                        content
                      );
                    }}
                  />
                )}
              </div>
            );
          })}
        </div>
      </div>
    );
  }
);
