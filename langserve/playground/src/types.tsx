export interface StreamCallback {
  onSuccess?: (ctx: { input: unknown; output: unknown }) => void;
  onError?: () => void;
  onStart?: (ctx: { input: unknown }) => void;
}
