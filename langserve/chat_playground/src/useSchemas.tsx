import { JsonSchema } from "@jsonforms/core";
import { compressToEncodedURIComponent } from "lz-string";
import { resolveApiUrl } from "./utils/url";
import { simplifySchema } from "./utils/simplifySchema";

import useSWR from "swr";
import defaults from "./utils/defaults";

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    CONFIG_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    INPUT_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    OUTPUT_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    FEEDBACK_ENABLED?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    PUBLIC_TRACE_LINK_ENABLED?: any;
  }
}

export function useFeedback() {
  return useSWR(["/feedback"], async () => {
    if (!import.meta.env.DEV && window.FEEDBACK_ENABLED) {
      return window.FEEDBACK_ENABLED === "true";
    }

    const response = await fetch(resolveApiUrl("/feedback"), {
      method: "HEAD",
    });
    return response.ok;
  });
}

export function usePublicTraceLink() {
  return useSWR(["/public_trace_link"], async () => {
    if (!import.meta.env.DEV && window.PUBLIC_TRACE_LINK_ENABLED) {
      return window.PUBLIC_TRACE_LINK_ENABLED === "true";
    }

    const response = await fetch(resolveApiUrl("/public_trace_link"), {
      method: "HEAD",
    });
    return response.ok;
  });
}

export function useConfigSchema() {
  return useSWR(["/config_schema"], async () => {
    let schema: JsonSchema | null = null;
    if (!import.meta.env.DEV && window.CONFIG_SCHEMA) {
      schema = await simplifySchema(window.CONFIG_SCHEMA);
    } else {
      const response = await fetch(resolveApiUrl(`/config_schema`));
      if (!response.ok) throw new Error(await response.text());

      const json = await response.json();
      schema = await simplifySchema(json);
    }

    if (schema == null) return null;
    return {
      schema,
      defaults: defaults(schema),
    };
  });
}

export function useInputSchema(configData?: unknown) {
  return useSWR(
    ["/input_schema", configData],
    async ([, configData]) => {
      // TODO: this won't work if we're already seeing a prefixed URL
      const prefix = configData
        ? `/c/${compressToEncodedURIComponent(JSON.stringify(configData))}`
        : "";

      let schema: JsonSchema | null = null;

      if (!prefix && !import.meta.env.DEV && window.INPUT_SCHEMA) {
        schema = await simplifySchema(window.INPUT_SCHEMA);
      } else {
        const response = await fetch(resolveApiUrl(`${prefix}/input_schema`));
        if (!response.ok) throw new Error(await response.text());

        const json = await response.json();
        schema = await simplifySchema(json);
      }

      if (schema == null) return null;
      return {
        schema,
        defaults: defaults(schema),
      };
    },
    { keepPreviousData: true }
  );
}

export function useOutputSchema(configData?: unknown) {
  return useSWR(
    ["/output_schema", configData],
    async ([, configData]) => {
      // TODO: this won't work if we're already seeing a prefixed URL
      const prefix = configData
        ? `/c/${compressToEncodedURIComponent(JSON.stringify(configData))}`
        : "";

      let schema: JsonSchema | null = null;

      if (!prefix && !import.meta.env.DEV && window.OUTPUT_SCHEMA) {
        schema = await simplifySchema(window.OUTPUT_SCHEMA);
      } else {
        const response = await fetch(resolveApiUrl(`${prefix}/output_schema`));
        if (!response.ok) throw new Error(await response.text());

        const json = await response.json();
        schema = await simplifySchema(json);
      }

      if (schema == null) return null;
      return {
        schema,
        defaults: defaults(schema),
      };
    },
    { keepPreviousData: true }
  );
}
