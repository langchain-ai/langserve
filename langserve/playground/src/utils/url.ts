export function resolveApiUrl(path: string) {
  let prefix = window.location.pathname.split("/playground")[0];
  if (prefix.endsWith("/")) prefix = prefix.slice(0, -1);
  return new URL(prefix + path, window.location.origin);
}
