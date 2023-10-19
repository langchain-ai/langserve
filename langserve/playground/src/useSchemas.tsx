import { useEffect, useState } from "react";
import { resolveApiUrl } from "./utils/url";
import { simplifySchema } from "./utils/simplifySchema";
import { JsonFormsCore } from "@jsonforms/core";
import { compressToEncodedURIComponent } from "lz-string";

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    CONFIG_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    INPUT_SCHEMA?: any;
  }
}

export function useSchemas(
  configData: Pick<JsonFormsCore, "data" | "errors"> & { defaults: boolean }
) {
  const [schemas, setSchemas] = useState({
    config: null,
    input: null,
  });

  useEffect(() => {
    async function save() {
      if (import.meta.env.DEV) {
        const [config, input] = await Promise.all([
          fetch(resolveApiUrl("/config_schema"))
            .then((r) => r.json())
            .then(simplifySchema),
          fetch(resolveApiUrl("/input_schema"))
            .then((r) => r.json())
            .then(simplifySchema),
        ]);
        setSchemas({ config, input });
      } else {
        setSchemas({
          config: window.CONFIG_SCHEMA
            ? await simplifySchema(window.CONFIG_SCHEMA)
            : null,
          input: window.INPUT_SCHEMA
            ? await simplifySchema(window.INPUT_SCHEMA)
            : null,
        });
      }
    }

    save();
  }, []);

  useEffect(() => {
    if (!configData.defaults) {
      fetch(
        resolveApiUrl(
          `c/${compressToEncodedURIComponent(
            JSON.stringify(configData.data)
          )}/input_schema`
        )
      )
        .then((r) => r.json())
        .then(simplifySchema)
        .then((input) => setSchemas((current) => ({ ...current, input })));
    }
  }, [configData]);

  return schemas;
}
