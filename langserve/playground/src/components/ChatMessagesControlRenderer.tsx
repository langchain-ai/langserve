import { withJsonFormsControlProps } from "@jsonforms/react";
import PlusIcon from "../assets/PlusIcon.svg?react";
import {
  rankWith,
  and,
  schemaMatches,
  Paths,
  isControl,
  JsonSchema,
} from "@jsonforms/core";
import { useStreamCallback } from "../useStreamCallback";
import { isJsonSchemaExtra } from "../utils/schema";
import { MessageFields, ChatMessageInput } from "./ChatMessageInput";
import { useEffect } from "react";

function checkItemSchema(schema: JsonSchema) {
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
}

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
        return schema.items.anyOf.every(checkItemSchema);
      }

      if ("oneOf" in schema.items && schema.items.oneOf != null) {
        return schema.items.oneOf.every(checkItemSchema);
      }

      return false;
    })
  )
);

export const ChatMessagesControlRenderer = withJsonFormsControlProps(
  (props) => {
    const data: Array<MessageFields> = props.data ?? [];

    useEffect(() => {
      if (!isJsonSchemaExtra(props.schema)) return;
      if (props.schema.extra.widget.type !== "chat") return;
      setTimeout(
        () =>
          props.handleChange(props.path, [
            ...data,
            { content: "", type: "human" },
          ]),
        10
      );
    }, []);

    useStreamCallback("onStart", () => {
      if (!isJsonSchemaExtra(props.schema)) return;
      if (props.schema.extra.widget.type !== "chat") return;
      props.handleChange(props.path, [...data, { content: "", type: "ai" }]);
    });

    useStreamCallback("onChunk", (_chunk, aggregatedState) => {
      if (!isJsonSchemaExtra(props.schema)) return;
      if (props.schema.extra.widget.type !== "chat") return;
      if (aggregatedState?.final_output !== undefined) {
        const msgPath = Paths.compose(props.path, `${data.length - 1}`);
        if (
          (aggregatedState.final_output as MessageFields)?.type ===
          "AIMessageChunk"
        ) {
          props.handleChange(
            Paths.compose(msgPath, "content"),
            (aggregatedState.final_output as MessageFields)?.content
          );
        } else if (typeof aggregatedState.final_output === "string") {
          props.handleChange(
            Paths.compose(msgPath, "content"),
            aggregatedState.final_output
          );
        }
      }
    });

    useStreamCallback("onSuccess", () => {
      if (!isJsonSchemaExtra(props.schema)) return;
      if (props.schema.extra.widget.type !== "chat") return;
      props.handleChange(props.path, [...data, { content: "", type: "human" }]);
    });

    useStreamCallback("onError", () => {
      if (data.length && data[data.length - 1].type === "ai") {
        props.handleChange(props.path, [...data.slice(0, -1)]);
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

            const handleChatMessageChange = (field: string, value: any) => {
              props.handleChange(Paths.compose(msgPath, field), value);
            };

            const handleChatMessageRemoval = () => {
              props.handleChange(
                props.path,
                data.filter((_, i) => i !== index)
              );
            };
            return (
              <ChatMessageInput
                message={message}
                handleChange={handleChatMessageChange}
                handleRemoval={handleChatMessageRemoval}
                path={props.path}
                key={index}
              ></ChatMessageInput>
            );
          })}
        </div>
      </div>
    );
  }
);
