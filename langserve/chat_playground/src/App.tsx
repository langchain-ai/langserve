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
  const isSupported = isLoading || (
    inputKeys.length === 1 &&
    inputProps[inputKeys[0]].type === "array" &&
    (
      outputDataSchema.anyOf?.find((option) => option.properties?.type?.enum?.includes("ai")) ||
      outputDataSchema.type === "string"
    )
  );
  return (
    <div className="flex items-center flex-col text-ls-black bg-background">
      <AppCallbackContext.Provider value={context}>
        {isSupported
          ? <ChatWindow
              startStream={startStream}
              stopStream={stopStream}
              inputKey={inputKeys[0]}
            ></ChatWindow>
          : <div className="h-[100vh] w-[100vw] flex justify-center items-center text-xl">
              <span className="text-center">
                The chat playground is only supported for chains that take a single array of messages as input
                <br/>
                and return either an AIMessage or a string.
                <br />
                <br />
                You can test this chain in the default LangServe playground instead. Please set <code>playground_type="default"</code>.
              </span>
            </div>}
      </AppCallbackContext.Provider>
    </div>
  );
}

export default App;
