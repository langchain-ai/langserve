export function getMessageContent(x: unknown) {
  if (typeof x === "string") return x;
  if (typeof x === "object" && x != null) {
    if ("content" in x && typeof x.content === "string") return x.content;
  }
  return null;
}
