import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { AutosizeTextarea } from "./AutosizeTextarea";
import TrashIcon from "../assets/TrashIcon.svg?react";
import CodeIcon from "../assets/CodeIcon.svg?react";
import ChatIcon from "../assets/ChatIcon.svg?react";

export interface MessageFields {
  content: string;
  additional_kwargs?: { [key: string]: unknown };
  name?: string;
  type?: string;
  role?: string;
}

export interface ChatMessageInputArgs {
  handleChange: (field: string, value: any) => void;
  handleRemoval: () => void;
  path: string;
  message: MessageFields;
};

function isOpenAiFunctionCall(
  x: unknown
): x is { name: string; arguments: string } {
  if (typeof x !== "object" || x == null) return false;
  if (!("name" in x) || typeof x.name !== "string") return false;
  if (!("arguments" in x) || typeof x.arguments !== "string") return false;
  return true;
}

export const ChatMessageInput = (props: ChatMessageInputArgs) => {
  const { message } = props;
  const isAiFunctionCall = isOpenAiFunctionCall(
    message.additional_kwargs?.function_call
  );
  const type = message.type ?? "chat"
  return (
    <div className="control group">
      <div className="flex items-start justify-between gap-2">
        <select
          className="-ml-1 min-w-[100px]"
          value={type}
          onChange={(e) => {
            props.handleChange(
              "type",
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
                      "additional_kwargs",
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
                      "additional_kwargs",
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
            onClick={props.handleRemoval}
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
              "role",
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
              "name",
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
                "additional_kwargs.function_call.name",
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
                "additional_kwargs.function_call.arguments",
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
              "content",
              content
            );
          }}
        />
      )}
    </div>
  );
}