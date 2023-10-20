import { withJsonFormsControlProps } from "@jsonforms/react";
import PlusIcon from "../assets/PlusIcon.svg?react";
import {
  rankWith,
  and,
  schemaMatches,
  Paths,
  isControl,
} from "@jsonforms/core";
import { AutosizeTextarea } from "./AutosizeTextarea";

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
        return schema.items.anyOf.every(
          (schema) =>
            schema.type === "object" &&
            (schema.title?.endsWith("Message") ||
              schema.title?.endsWith("MessageChunk"))
        );
      }

      return false;
    })
  )
);

export const ChatMessageControlRenderer = withJsonFormsControlProps((props) => {
  const data: Array<{
    content: string;
    additional_kwargs?: { [key: string]: unknown };
    name?: string;
    type?: string;

    role?: string;
  }> = props.data ?? [];

  return (
    <div className="control">
      <div className="flex items-center justify-between">
        <label className="text-xs uppercase font-semibold text-ls-gray-100">
          {props.label}
        </label>
        <button
          className="p-1 rounded-full"
          onClick={() => {
            props.handleChange(props.path, [
              ...data,
              { content: "", type: "ai" },
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
          return (
            <div className="control" key={index}>
              <select
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

              {/* {type === "chat" && <input placeholder="Role" />}

              {type === "function" && <input placeholder="Function Name" />} */}

              <AutosizeTextarea
                value={message.content}
                onChange={(content) => {
                  props.handleChange(
                    Paths.compose(msgPath, "content"),
                    content
                  );
                }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
});
