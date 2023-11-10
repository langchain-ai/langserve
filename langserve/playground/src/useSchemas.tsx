import { useEffect, useState } from "react";
import { resolveApiUrl } from "./utils/url";
import { simplifySchema } from "./utils/simplifySchema";
import { JsonFormsCore } from "@jsonforms/core";
import { compressToEncodedURIComponent } from "lz-string";
import { useDebounce } from "use-debounce";

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    CONFIG_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    INPUT_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    FEEDBACK_ENABLED?: any;
  }
}

export function useSchemas(
  configData: Pick<JsonFormsCore, "data" | "errors"> & { defaults: boolean }
) {
  const [schemas, setSchemas] = useState<{
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    config: null | any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    input: null | any;

    feedbackEnabled: null | boolean;
  }>({
    config: null,
    input: null,
    feedbackEnabled: null,
  });

  useEffect(() => {
    async function save() {
      if (import.meta.env.DEV) {
        const [config, input, feedbackEnabled] = await Promise.all([
          fetch(resolveApiUrl("/config_schema"))
            .then((r) => r.json())
            .then(simplifySchema),
          fetch(resolveApiUrl("/input_schema"))
            .then((r) => r.json())
            .then(simplifySchema),
          fetch(resolveApiUrl("/feedback"), { method: "HEAD" }).then((a) =>
            a.ok ? "1" : "0"
          ),
        ]);
        setSchemas({ config, input, feedbackEnabled: feedbackEnabled === "1" });
      } else {
        setSchemas({
          config: window.CONFIG_SCHEMA
            ? await simplifySchema(window.CONFIG_SCHEMA)
            : null,
          input: window.INPUT_SCHEMA
            ? await simplifySchema(window.INPUT_SCHEMA)
            : null,
          feedbackEnabled: window.FEEDBACK_ENABLED === "1",
        });
      }
    }

    save();
  }, []);

  const [debouncedConfigData] = useDebounce(configData, 500);

  useEffect(() => {
    if (!debouncedConfigData.defaults) {
      fetch(
        resolveApiUrl(
          `/c/${compressToEncodedURIComponent(
            JSON.stringify(debouncedConfigData.data)
          )}/input_schema`
        )
      )
        .then((r) => r.json())
        .then(simplifySchema)
        .then((input) => setSchemas((current) => ({ ...current, input })))
        .catch(() => {}); // ignore errors, eg. due to incomplete config
    }
  }, [debouncedConfigData]);

  return schemas;
}
