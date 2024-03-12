import "./App.css";

import { ChatWindow } from "./components/ChatWindow";
import { AppCallbackContext, useAppStreamCallbacks } from "./useStreamCallback";
import { useInputSchema, useOutputSchema } from "./useSchemas";
import { useStreamLog } from "./useStreamLog";

export function App() {
  const { context, callbacks } = useAppStreamCallbacks();
  const { startStream, stopStream } = useStreamLog(callbacks);
  const inputSchema = useInputSchema({});
  const outputSchema = useOutputSchema({});
  const inputProps = inputSchema?.data?.schema?.properties;
  const outputDataSchema = outputSchema?.data?.schema;
  const isLoading = inputProps === undefined || outputDataSchema === undefined;
  const inputKeys = Object.keys(inputProps ?? {});
  const inputSchemaSupported = (
    inputKeys.length === 1 &&
    inputProps?.[inputKeys[0]].type === "array"
  ) || (
    inputKeys.length === 2 && (
      (
        inputProps?.[inputKeys[0]].type === "array" ||
        inputProps?.[inputKeys[1]].type === "string"
      ) || (
        inputProps?.[inputKeys[0]].type === "string" ||
        inputProps?.[inputKeys[1]].type === "array" 
      )
    )
  );
  const outputSchemaSupported = (
    outputDataSchema?.anyOf?.find((option) => option.properties?.type?.enum?.includes("ai")) ||
    outputDataSchema?.type === "string"
  );
  const isSupported = isLoading || (inputSchemaSupported && outputSchemaSupported);
  return (
    <div className="flex items-center flex-col text-ls-black bg-background">
      <AppCallbackContext.Provider value={context}>
        {isSupported
          ? <ChatWindow
              startStream={startStream}
              stopStream={stopStream}
              messagesInputKey={inputProps?.[inputKeys[0]].type === "array" ? inputKeys[0] : inputKeys[1]}
              inputKey={inputProps?.[inputKeys[0]].type === "string" ? inputKeys[0] : inputKeys[1]}
            ></ChatWindow>
          : <div className="h-[100vh] w-[100vw] flex justify-center items-center text-xl p-16">
              <span>
                The chat playground is only supported for chains that take one of the following as input:
                <ul className="mt-8 list-disc ml-6">
                  <li>
                    a dict with a single key containing a list of messages
                  </li>
                  <li>
                    a dict with two keys: one a string input, one an list of messages
                  </li>
                </ul>
                <br />
                and which return either an <code>AIMessage</code> or a string.
                <br />
                <br />
                You can test this chain in the default LangServe playground instead.
                <br />
                <br />
                To use the default playground, set <code>playground_type="default"</code> when adding the route in your backend.
              </span>
            </div>}
      </AppCallbackContext.Provider>
    </div>
  );
}

export default App;
