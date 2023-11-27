import "./App.css";

import { useEffect, useRef, useState } from "react";

import ShareIcon from "./assets/ShareIcon.svg?react";

import { useConfigSchema, useFeedback, useInputSchema } from "./useSchemas";
import { useStreamLog } from "./useStreamLog";
import { AppCallbackContext, useAppStreamCallbacks } from "./useStreamCallback";
import { JsonSchema } from "@jsonforms/core";
import { ShareDialog } from "./components/ShareDialog";
import { IntermediateSteps } from "./components/IntermediateSteps";
import { StreamOutput } from "./components/StreamOutput";
import { ConfigValue, SectionConfigure } from "./sections/SectionConfigure";
import { InputValue, SectionInputs } from "./sections/SectionInputs";
import { SubmitButton } from "./components/SubmitButton";
import { useDebounce } from "use-debounce";
import { cn } from "./utils/cn";
import { CorrectnessFeedback } from "./components/feedback/CorrectnessFeedback";
import { getStateFromUrl } from "./utils/url";

function InputPlayground(props: {
  configSchema: { schema: JsonSchema; defaults: unknown };
  inputSchema: { schema: JsonSchema; defaults: unknown };

  configData: ConfigValue;

  startStream: (input: unknown, config: unknown) => void;
  stopStream: (() => void) | undefined;

  children?: React.ReactNode;
}) {
  const [inputData, setInputData] = useState<InputValue>({
    data: props.inputSchema.defaults,
    errors: [],
  });

  const submitRef = useRef<(() => void) | null>(null);
  submitRef.current = () => {
    if (
      !props.stopStream &&
      (!!inputData.errors?.length || !!props.configData.errors?.length)
    ) {
      return;
    }

    if (props.stopStream) {
      props.stopStream();
    } else {
      props.startStream(inputData.data, props.configData.data);
    }
  };

  useEffect(() => {
    window.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        submitRef.current?.();
      }
    });
  }, []);

  const isSendDisabled =
    !props.stopStream &&
    (!!inputData.errors?.length || !!props.configData.errors?.length);

  return (
    <>
      <SectionInputs
        input={props.inputSchema.schema}
        value={inputData}
        onChange={(input) => setInputData(input)}
      />

      {props.children}

      <div className="flex-grow md:hidden" />

      <div className="gap-4 grid grid-cols-2 sticky -mx-4 px-4 py-4 bottom-0 bg-background md:static md:bg-transparent">
        <div className="md:hidden absolute inset-x-0 bottom-full h-5 bg-gradient-to-t from-black/5 to-black/0" />

        <ShareDialog config={props.configData.data}>
          <button
            type="button"
            className="px-4 py-3 gap-3 font-medium border border-divider-700 rounded-full flex items-center justify-center hover:bg-divider-500/50 active:bg-divider-500 transition-colors"
          >
            <ShareIcon className="flex-shrink-0" /> <span>Share</span>
          </button>
        </ShareDialog>

        <SubmitButton
          disabled={isSendDisabled}
          onSubmit={submitRef.current}
          isLoading={!!props.stopStream}
        />
      </div>
    </>
  );
}

function ConfigPlayground(props: {
  configSchema: {
    schema: JsonSchema;
    defaults: unknown;
  };
}) {
  const urlState = getStateFromUrl(window.location.href);
  const [configData, setConfigData] = useState<ConfigValue>({
    data: urlState.configFromUrl ?? props.configSchema.defaults,
    errors: [],
    defaults: true,
  });

  const feedback = useFeedback();

  // input schema is derived from config data
  const [debouncedConfigData, debounceState] = useDebounce(
    configData.data,
    500
  );

  const inputSchema = useInputSchema(
    debouncedConfigData !== props.configSchema.defaults
      ? debouncedConfigData
      : undefined
  );

  const { context, callbacks } = useAppStreamCallbacks();
  const { startStream, stopStream, latest } = useStreamLog(callbacks);

  return (
    <AppCallbackContext.Provider value={context}>
      <SectionConfigure
        config={props.configSchema.schema}
        value={configData}
        onChange={setConfigData}
      />

      <div
        className={cn(
          "flex flex-col flex-grow gap-4 w-full transition-opacity",
          (inputSchema.isLoading || debounceState.isPending()) &&
            "opacity-50 pointer-events-none"
        )}
      >
        {inputSchema.error != null ? (
          <div className="bg-background rounded-xl">
            <div className="bg-red-500/10 text-red-700 dark:text-red-300 rounded-xl p-3">
              {inputSchema.error.toString()}
            </div>
          </div>
        ) : (
          <>
            {inputSchema.data != null ? (
              <InputPlayground
                configSchema={props.configSchema}
                inputSchema={inputSchema.data}
                configData={configData}
                startStream={startStream}
                stopStream={stopStream}
              >
                {latest && (
                  <div className="flex flex-col gap-3">
                    <h2 className="text-xl font-semibold">Output</h2>
                    <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-2xl bg-background text-lg whitespace-pre-wrap break-words relative group">
                      <StreamOutput streamed={latest.streamed_output} />

                      {feedback.data && latest.id ? (
                        <div className="absolute right-4 top-4 flex items-center gap-2 transition-opacity opacity-0 focus-within:opacity-100 group-hover:opacity-100">
                          <CorrectnessFeedback
                            key={latest.id}
                            runId={latest.id}
                          />
                        </div>
                      ) : null}
                    </div>
                    <IntermediateSteps
                      latest={latest}
                      feedbackEnabled={!!feedback.data}
                    />
                  </div>
                )}
              </InputPlayground>
            ) : null}
          </>
        )}
      </div>
    </AppCallbackContext.Provider>
  );
}

function Playground() {
  const configSchema = useConfigSchema();
  if (configSchema.isLoading) return null;

  if (configSchema.error != null) {
    return (
      <div className="bg-background rounded-xl">
        <div className="bg-red-500/10 text-red-700 dark:text-red-300 rounded-xl p-3">
          {configSchema.error.toString()}
        </div>
      </div>
    );
  }
  if (configSchema.data == null) return "No config schema found";
  return <ConfigPlayground configSchema={configSchema.data} />;
}

export function App() {
  return (
    <div className="flex items-center flex-col text-ls-black bg-gradient-to-b from-[#F9FAFB] to-[#EFF8FF] min-h-[100dvh] dark:from-[#0C111C] dark:to-[#0C111C]">
      <div className="flex flex-col flex-grow gap-4 px-4 pt-6 max-w-[800px] w-full">
        <h1 className="text-2xl text-left">
          <strong>🦜 LangServe</strong> Playground
        </h1>
        <Playground />
      </div>
    </div>
  );
}

export default App;
