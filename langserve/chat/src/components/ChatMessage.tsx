import ThumbsDownIcon from "../assets/ThumbsDownIcon.svg?react"
import CheckCircleIcon2 from "../assets/CheckCircleIcon2.svg?react"

export type ChatMessageType = {
  role: "human" | "ai" | "function" | "tool" | "system";
  content: string;
}

export function ChatMessage(props: {message: ChatMessageType}) {
  const { content, role } = props.message;
  return (
    <div className="mb-8">
      <p className="font-medium text-transform uppercase mb-2">{role}</p>
      <p>{content}</p>
      {role === "ai" && (
        <div className="flex mt-4">
          <CheckCircleIcon2 className="h-6 w-6 text-gray-400 mr-2 stroke-1 cursor-pointer" />
          <ThumbsDownIcon className="h-6 w-6 text-gray-400 stroke-1 cursor-pointer" />
        </div>)
      }
    </div> 
  )
};
