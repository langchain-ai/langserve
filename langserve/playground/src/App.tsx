import React, { useEffect, useState } from "react";
import { Card, Button, Stack, createTheme, ThemeProvider } from "@mui/material";
import defaults from "json-schema-defaults";
import { JsonForms } from "@jsonforms/react";
import {
  materialRenderers,
  materialCells,
} from "@jsonforms/material-renderers";

import { useSchemas } from "./useSchemas";
import "./App.css";
import { useStreamLog } from "./useStreamLog";
import { JsonFormsCore } from "@jsonforms/core";

function str(o: unknown): React.ReactNode {
  return typeof o === "object"
    ? JSON.stringify(o, null, 2)
    : (o as React.ReactNode);
}

const inputMultilineTheme = createTheme({
  components: { MuiInput: { defaultProps: { multiline: true } } },
});

function App() {
  // store form state
  const [configData, setConfigData] = useState<
    Pick<JsonFormsCore, "data" | "errors">
  >({ data: {}, errors: [] });
  const [inputData, setInputData] = useState<
    Pick<JsonFormsCore, "data" | "errors">
  >({ data: {}, errors: [] });
  // fetch input and config schemas from the server
  const schemas = useSchemas();
  // apply defaults defined in each schema
  useEffect(() => {
    if (schemas.config) {
      setConfigData({ data: defaults(schemas.config), errors: [] });
      setInputData({ data: defaults(schemas.input), errors: [] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemas.config]);
  // the runner
  const { startStream, stopStream, latest } = useStreamLog();

  return schemas.config && schemas.input ? (
    <>
      <h1>Playground</h1>
      <Card sx={{ padding: 2 }}>
        <h2>Configuration</h2>
        <JsonForms
          schema={schemas.config}
          data={configData.data}
          renderers={materialRenderers}
          cells={materialCells}
          onChange={({ data, errors }) =>
            data ? setConfigData({ data, errors }) : undefined
          }
        />
        {!!configData.errors?.length && (
          <>
            <h3>Validation Errors</h3>
            {configData.errors?.map((e, i) => (
              <p key={i}>{e.message}</p>
            ))}
          </>
        )}
      </Card>

      <Card sx={{ marginTop: 2, padding: 2 }}>
        <h2>Inputs</h2>
        <ThemeProvider theme={inputMultilineTheme}>
          <JsonForms
            schema={schemas.input}
            data={inputData.data}
            renderers={materialRenderers}
            cells={materialCells}
            onChange={({ data, errors }) => setInputData({ data, errors })}
          />
          {!!inputData.errors?.length && (
            <>
              <h3>Validation Errors</h3>
              {inputData.errors?.map((e, i) => (
                <p key={i}>{e.message}</p>
              ))}
            </>
          )}
        </ThemeProvider>
      </Card>
      <Stack direction="row" spacing={2} sx={{ marginTop: 2 }}>
        <Button
          disabled={
            !!inputData.errors?.length ||
            !!configData.errors?.length ||
            !!stopStream
          }
          variant="contained"
          color="primary"
          onClick={() => startStream(inputData.data, configData.data)}
        >
          {stopStream ? "Running..." : "Run"}
        </Button>
        <Button
          disabled={!stopStream}
          variant="contained"
          color="secondary"
          onClick={stopStream}
        >
          Stop
        </Button>
      </Stack>
      {latest && (
        <Card sx={{ marginTop: 2, padding: 2 }}>
          <h2>Output</h2>
          {latest.streamed_output.map(str).join("") ?? "..."}
        </Card>
      )}
      {latest && (
        <Card sx={{ marginTop: 2, padding: 2 }}>
          <h2>Intermediate Steps</h2>
          {Object.values(latest.logs).map((log) => (
            <div key={log.id}>
              <h3>{log.name}</h3>
              <p>{log.start_time}</p>
              <pre>{str(log.final_output) ?? "..."}</pre>
            </div>
          ))}
        </Card>
      )}
    </>
  ) : null;
}

export default App;
