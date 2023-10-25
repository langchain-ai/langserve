export function resolveApiUrl(path: string) {
  if (import.meta.env.DEV) {
    return new URL(path, "http://127.0.0.1:8000");
  }

  const prefix = window.location.pathname.split("/playground")[0];
  return new URL(prefix + path, window.location.origin);
}
