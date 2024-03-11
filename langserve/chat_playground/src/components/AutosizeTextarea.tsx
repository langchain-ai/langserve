import { Ref } from "react";
import { cn } from "../utils/cn";

const COMMON_CLS = cn(
  "text-lg col-[1] row-[1] m-0 resize-none overflow-hidden whitespace-pre-wrap break-words border-none bg-transparent p-0"
);

export function AutosizeTextarea(props: {
  id?: string;
  inputRef?: Ref<HTMLTextAreaElement>;
  value?: string | null | undefined;
  placeholder?: string;
  className?: string;
  onChange?: (e: string) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  onKeyDown?: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  autoFocus?: boolean;
  readOnly?: boolean;
  cursorPointer?: boolean;
  disabled?: boolean;
  fullHeight?: boolean;
}) {
  return (
    <div className={cn("grid w-full", props.className) + (props.fullHeight ? "" : " max-h-80 overflow-auto")}>
      <textarea
        ref={props.inputRef}
        id={props.id}
        className={cn(
          COMMON_CLS,
          "text-transparent caret-black"
        )}
        disabled={props.disabled}
        value={props.value ?? ""}
        rows={1}
        onChange={(e) => {
          const target = e.target as HTMLTextAreaElement;
          props.onChange?.(target.value);
        }}
        onFocus={props.onFocus}
        onBlur={props.onBlur}
        placeholder={props.placeholder}
        readOnly={props.readOnly}
        autoFocus={props.autoFocus && !props.readOnly}
        onKeyDown={props.onKeyDown}
      />
      <div
        aria-hidden
        className={cn(COMMON_CLS, "pointer-events-none select-none")}
      >
        {props.value}{" "}
      </div>
    </div>
  );
}
