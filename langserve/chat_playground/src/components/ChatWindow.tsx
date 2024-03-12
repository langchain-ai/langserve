import { useState, useRef } from "react";
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import { AutosizeTextarea } from "./AutosizeTextarea";
import {
  ChatMessage,
  type ChatMessageType,
  type ChatMessageBody,
} from "./ChatMessage";
import { ShareDialog } from "./ShareDialog";
import { useStreamCallback } from "../useStreamCallback";

import ArrowUp from "../assets/ArrowUp.svg?react";
import CircleSpinIcon from "../assets/CircleSpinIcon.svg?react";
import EmptyState from "../assets/EmptyState.svg?react";
import LangServeLogo from "../assets/LangServeLogo.svg?react";
import { useFeedback, usePublicTraceLink } from "../useSchemas";

export type AIMessage = {
  content: string;
  type: "AIMessage" | "AIMessageChunk";
  name?: string;
  additional_kwargs?: { [key: string]: unknown };
}

export function isAIMessage(x: unknown): x is AIMessage {
  return x != null &&
    typeof (x as AIMessage).content === "string" &&
    ["AIMessageChunk", "AIMessage"].includes((x as AIMessage).type);
}

export function ChatWindow(props: {
  startStream: (input: unknown, config: unknown) => Promise<void>;
  stopStream: (() => void) | undefined;
  messagesInputKey: string;
  inputKey?: string;
}) {
  const { startStream, messagesInputKey, inputKey } = props;

  const [currentInputValue, setCurrentInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<ChatMessageBody[]>([]);
  const messageInputRef = useRef<HTMLTextAreaElement>(null);

  const feedbackEnabled = useFeedback()
  const publicTraceLinksEnabled = usePublicTraceLink();

  const submitMessage = () => {
    const submittedValue = currentInputValue;
    if (submittedValue.length === 0 || isLoading) {
      return;
    }
    setIsLoading(true);
    const newMessages = [
      ...messages,
      { type: "human", content: submittedValue } as const
    ];
    setMessages(newMessages);
    setCurrentInputValue("");
    // TODO: Add config schema support
    if (inputKey === undefined) {
      startStream({ [messagesInputKey]: newMessages }, {});
    } else {
      startStream({
        [messagesInputKey]: newMessages.slice(0, -1),
        [inputKey]: newMessages[newMessages.length - 1].content
      }, {});
    }
  };

  const regenerateMessages = () => {
    if (isLoading) {
      return;
    }
    setIsLoading(true);
    // TODO: Add config schema support
    if (inputKey === undefined) {
      startStream({ [messagesInputKey]: messages }, {});
    } else {
      startStream({
        [messagesInputKey]: messages.slice(0, -1),
        [inputKey]: messages[messages.length - 1].content
      }, {});
    }
  };
  
  useStreamCallback("onStart", () => {
    setMessages((prevMessages) => [
      ...prevMessages,
      { type: "ai", content: "" },
    ]);
  });
  useStreamCallback("onChunk", (_chunk, aggregatedState) => {
    const finalOutput = aggregatedState?.final_output;
    if (typeof finalOutput === "string") {
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        { type: "ai", content: finalOutput, runId: aggregatedState?.id }
      ]); 
    } else if (isAIMessage(finalOutput)) {
      setMessages((prevMessages) => [
        ...prevMessages.slice(0, -1),
        { type: "ai", content: finalOutput.content, runId: aggregatedState?.id }
      ]);
    }
  });
  useStreamCallback("onSuccess", () => {
    setIsLoading(false);
  });
  useStreamCallback("onError", (e) => {
    setIsLoading(false);
    toast(e.message + "\nCheck your backend logs for errors.", { hideProgressBar: true });
    setCurrentInputValue(messages[messages.length - 2]?.content);
    setMessages((prevMessages) => [
      ...prevMessages.slice(0, -2),
    ]);
  });

  return (
    <div className="flex flex-col h-screen w-screen">
      <nav className="flex items-center justify-between p-8">
        <div className="flex items-center">
          <LangServeLogo />
          <span className="ml-1">Playground</span>
        </div>
        <div className="flex items-center space-x-4">
          <ShareDialog config={{}}>
            <button
              type="button"
              className="px-3 py-1 border rounded-full px-8 py-2 share-button"
            >
              <span>Share</span>
            </button>
          </ShareDialog>
        </div>
      </nav>
      <div className="flex-grow flex flex-col items-center justify-center mt-8">
        {messages.length > 0 ? (
          <div className="flex flex-col-reverse basis-0 overflow-auto flex-re grow max-w-[640px] w-[640px]">
            {messages.map((message, i) => {
              return (
                <ChatMessage
                  message={message}
                  key={i}
                  isLoading={isLoading}
                  onError={(e: any) => toast(e.message, { hideProgressBar: true })}
                  feedbackEnabled={feedbackEnabled.data}
                  publicTraceLinksEnabled={publicTraceLinksEnabled.data}
                  isFinalMessage={i === messages.length - 1}
                  onRemove={() => setMessages(
                    (previousMessages) => [...previousMessages.slice(0, i), ...previousMessages.slice(i + 1)]
                  )}
                  onTypeChange={(newValue) => {
                    setMessages(
                      (previousMessages) => [
                        ...previousMessages.slice(0, i),
                        {...message, type: newValue as ChatMessageType},
                        ...previousMessages.slice(i + 1)
                      ]
                    )
                  }}
                  onChange={(newValue) => {
                    setMessages(
                      (previousMessages) => [
                        ...previousMessages.slice(0, i),
                        {...message, content: newValue},
                        ...previousMessages.slice(i + 1)
                      ]
                    );
                  }}
                  onRegenerate={() => regenerateMessages()}
                ></ChatMessage>
              );
            }).reverse()}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center">
            <EmptyState />
            <h1 className="text-lg">Start testing your application</h1>
          </div>
        )}
      </div>
      <div className="m-16 mt-4 flex justify-center">
        <div className="flex items-center p-3 rounded-[48px] border shadow-sm max-w-[768px] grow" onClick={() => messageInputRef.current?.focus()}>
          <AutosizeTextarea
            inputRef={messageInputRef}
            className="flex-grow mr-4 ml-8 border-none focus:ring-0 py-2 cursor-text"
            placeholder="Send a message..."
            value={currentInputValue}
            onChange={(newValue) => {
              setCurrentInputValue(newValue);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitMessage();
              }
            }}
          />
          <button
            className={"flex items-center justify-center px-3 py-1 rounded-[40px] " + (isLoading ? "" : currentInputValue.length > 0 ? "bg-button-green" : "bg-button-green-disabled")}
            onClick={(e) => {
              e.preventDefault();
              submitMessage();
            }}
          >
            {isLoading 
              ? <CircleSpinIcon className="animate-spin w-5 h-5 text-background fill-background" />
              : <ArrowUp className="mx-2 my-2 h-5 w-5 stroke-white" />}
          </button>
        </div>
      </div>
      <ToastContainer />
    </div>
  )
}
