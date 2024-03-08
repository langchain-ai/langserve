import "./App.css";

import { ChatWindow } from "./components/ChatWindow";
import { AppCallbackContext, useAppStreamCallbacks } from "./useStreamCallback";
import { useStreamLog } from "./useStreamLog";

export function App() {
  const { context, callbacks } = useAppStreamCallbacks();
  const { startStream, stopStream } = useStreamLog(callbacks);

  return (
    <div className="flex items-center flex-col text-ls-black bg-background">
      <AppCallbackContext.Provider value={context}>
        <ChatWindow startStream={startStream} stopStream={stopStream}></ChatWindow>
      </AppCallbackContext.Provider>
    </div>
  );
}

export default App;
