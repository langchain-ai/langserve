import { useEffect, useState } from "react";

declare global {
  interface Window {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    CONFIG_SCHEMA?: any;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    INPUT_SCHEMA?: any;
  }
}

export function useSchemas() {
  const [schemas, setSchemas] = useState({
    config: null,
    input: null,
  });

  useEffect(() => {
    async function save() {
      if (import.meta.env.DEV) {
        const [config, input] = await Promise.all([
          fetch("http://localhost:8003/config_schema").then((r) => r.json()),
          fetch("http://localhost:8003/input_schema").then((r) => r.json()),
        ]);
        setSchemas({ config, input });
      } else {
        setSchemas({
          config: window.CONFIG_SCHEMA ?? null,
          input: window.INPUT_SCHEMA ?? null,
        });
      }
    }

    save();
  }, []);

  return schemas;
}
