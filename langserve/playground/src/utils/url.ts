import { decompressFromEncodedURIComponent } from "lz-string";

export function getStateFromUrl(path: string) {
  let configFromUrl = null;
  let basePath = path;
  if (basePath.endsWith("/")) {
    basePath = basePath.slice(0, -1);
  }

  if (basePath.endsWith("/playground")) {
    basePath = basePath.slice(0, -"/playground".length);
  }

  // check if we can omit the last segment
  const [configHash, c, ...rest] = basePath.split("/").reverse();
  if (c === "c") {
    basePath = rest.reverse().join("/");
    try {
      configFromUrl = JSON.parse(decompressFromEncodedURIComponent(configHash));
    } catch (error) {
      console.error(error);
    }
  }
  return { basePath, configFromUrl };
}

export function resolveApiUrl(path: string) {
  const { basePath } = getStateFromUrl(window.location.href);
  let prefix = new URL(basePath).pathname;
  if (prefix.endsWith("/")) prefix = prefix.slice(0, -1);

  return new URL(prefix + path, basePath);
}
