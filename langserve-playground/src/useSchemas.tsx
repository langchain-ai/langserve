import { useEffect, useState } from "react";

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
          config: window.CONFIG_SCHEMA,
          input: window.INPUT_SCHEMA,
        });
      }
    }

    save();
  }, []);

  return schemas;
}
