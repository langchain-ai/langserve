import "./App.css";

import { ChatWindow } from "./components/ChatWindow";
import { AppCallbackContext, useAppStreamCallbacks } from "./useStreamCallback";
import { useInputSchema } from "./useSchemas";
import { useStreamLog } from "./useStreamLog";
import { resolveApiUrl } from "./utils/url";

export function App() {
  const { context, callbacks } = useAppStreamCallbacks();
  const { startStream, stopStream } = useStreamLog(callbacks);
  const inputSchema = useInputSchema({});
  const inputProps = inputSchema?.data?.schema?.properties;
  const isLoading = inputProps === undefined;
  const inputKeys = Object.keys(inputProps ?? {});
  const isSupported = isLoading || (inputKeys.length === 1 && inputProps[inputKeys[0]].type === "array");
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
                The chat playground is only supported for chains that take a single array of messages as input.
                <br/>
                You can test this chain in the standard <a href={resolveApiUrl("/playground").toString()}>LangServe playground</a>.
              </span>
            </div>}
      </AppCallbackContext.Provider>
    </div>
  );
}

export default App;
