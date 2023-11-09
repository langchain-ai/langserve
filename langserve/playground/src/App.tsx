import "./App.css";

import { useEffect, useMemo, useRef, useState } from "react";
import defaults from "./utils/defaults";
import { JsonForms } from "@jsonforms/react";
import {
  materialAllOfControlTester,
  MaterialAllOfRenderer,
  MaterialObjectRenderer,
  materialOneOfControlTester,
  MaterialOneOfRenderer,
} from "@jsonforms/material-renderers";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import relativeDate from "dayjs/plugin/relativeTime";
import SendIcon from "./assets/SendIcon.svg?react";
import ShareIcon from "./assets/ShareIcon.svg?react";
import { compressToEncodedURIComponent } from "lz-string";

import {
  BooleanCell,
  DateCell,
  DateTimeCell,
  EnumCell,
  IntegerCell,
  NumberCell,
  SliderCell,
  TimeCell,
  booleanCellTester,
  dateCellTester,
  dateTimeCellTester,
  enumCellTester,
  integerCellTester,
  numberCellTester,
  sliderCellTester,
  textAreaCellTester,
  textCellTester,
  timeCellTester,
  vanillaRenderers,
  InputControl,
} from "@jsonforms/vanilla-renderers";
import { useSchemas } from "./useSchemas";
import { useStreamLog } from "./useStreamLog";
import { StreamCallback } from "./types";
import { AppCallbackContext } from "./useStreamCallback";
import {
  JsonFormsCore,
  RankedTester,
  rankWith,
  and,
  uiTypeIs,
  schemaMatches,
  schemaTypeIs,
} from "@jsonforms/core";
import CustomArrayControlRenderer, {
  materialArrayControlTester,
} from "./components/CustomArrayControlRenderer";
import CustomTextAreaCell from "./components/CustomTextAreaCell";
import JsonTextAreaCell from "./components/JsonTextAreaCell";
import { getStateFromUrl, ShareDialog } from "./components/ShareDialog";
import {
  chatMessagesTester,
  ChatMessagesControlRenderer,
} from "./components/ChatMessagesControlRenderer";
import {
  ChatMessageTuplesControlRenderer,
  chatMessagesTupleTester,
} from "./components/ChatMessageTuplesControlRenderer";
import {
  fileBase64Tester,
  FileBase64ControlRenderer,
} from "./components/FileBase64Tester";
import { IntermediateSteps } from "./components/IntermediateSteps";
import { StreamOutput } from "./components/StreamOutput";
import {
  customAnyOfTester,
  CustomAnyOfRenderer,
} from "./components/CustomAnyOfRenderer";
import { cn } from "./utils/cn";

dayjs.extend(relativeDate);
dayjs.extend(utc);

const isObjectWithPropertiesControl = rankWith(
  2,
  and(
    uiTypeIs("Control"),
    schemaTypeIs("object"),
    schemaMatches((schema) =>
      Object.prototype.hasOwnProperty.call(schema, "properties")
    )
  )
);

const isObject = rankWith(1, and(uiTypeIs("Control"), schemaTypeIs("object")));
const isElse = rankWith(1, and(uiTypeIs("Control")));

export const renderers = [
  ...vanillaRenderers,

  // use material renderers to handle objects and json schema references
  // they should yield the rendering to simpler cells
  { tester: isObjectWithPropertiesControl, renderer: MaterialObjectRenderer },
  { tester: materialAllOfControlTester, renderer: MaterialAllOfRenderer },
  { tester: materialOneOfControlTester, renderer: MaterialOneOfRenderer },

  { tester: customAnyOfTester, renderer: CustomAnyOfRenderer },

  // custom renderers
  { tester: materialArrayControlTester, renderer: CustomArrayControlRenderer },
  { tester: isObject, renderer: InputControl },
  { tester: chatMessagesTester, renderer: ChatMessagesControlRenderer },
  {
    tester: chatMessagesTupleTester,
    renderer: ChatMessageTuplesControlRenderer,
  },
  { tester: fileBase64Tester, renderer: FileBase64ControlRenderer },
];

const nestedArrayControlTester: RankedTester = rankWith(1, (_, jsonSchema) => {
  return jsonSchema.type === "array";
});

export const cells = [
  { tester: booleanCellTester, cell: BooleanCell },
  { tester: dateCellTester, cell: DateCell },
  { tester: dateTimeCellTester, cell: DateTimeCell },
  { tester: enumCellTester, cell: EnumCell },
  { tester: integerCellTester, cell: IntegerCell },
  { tester: numberCellTester, cell: NumberCell },
  { tester: sliderCellTester, cell: SliderCell },
  { tester: textAreaCellTester, cell: CustomTextAreaCell },
  { tester: textCellTester, cell: CustomTextAreaCell },
  { tester: timeCellTester, cell: TimeCell },
  { tester: nestedArrayControlTester, cell: CustomArrayControlRenderer },
  { tester: isElse, cell: JsonTextAreaCell },
];

function App() {
  const [isEmbedded] = useState(() =>
    window.location.search.includes("embeded=true")
  );

  // it is possible that defaults are being applied _after_
  // the initial update message has been sent from the parent window
  // so we store the initial config data in a ref
  const initConfigData = useRef<JsonFormsCore["data"]>(null);

  // store form state
  const [configData, setConfigData] = useState<
    Pick<JsonFormsCore, "data" | "errors"> & { defaults: boolean }
  >({ data: {}, errors: [], defaults: true });

  const [inputData, setInputData] = useState<
    Pick<JsonFormsCore, "data" | "errors">
  >({ data: null, errors: [] });
  // fetch input and config schemas from the server
  const schemas = useSchemas(configData);
  // apply defaults defined in each schema
  useEffect(() => {
    if (schemas.config) {
      const state = getStateFromUrl(window.location.href);
      setConfigData({
        data:
          state.configFromUrl ??
          initConfigData.current ??
          defaults(schemas.config),
        errors: [],
        defaults: true,
      });

      setInputData({ data: defaults(schemas.input), errors: [] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemas.config]);

  // callbacks handling
  const callbacks = useRef<{
    onStart: Exclude<StreamCallback["onStart"], undefined>[];
    onSuccess: Exclude<StreamCallback["onSuccess"], undefined>[];
    onError: Exclude<StreamCallback["onError"], undefined>[];
  }>({ onStart: [], onSuccess: [], onError: [] });

  // the runner
  const { startStream, stopStream, latest } = useStreamLog({
    onStart(...args) {
      for (const callback of callbacks.current.onStart) {
        callback(...args);
      }
    },
    onSuccess(...args) {
      for (const callback of callbacks.current.onSuccess) {
        callback(...args);
      }
    },
    onError(...args) {
      for (const callback of callbacks.current.onError) {
        callback(...args);
      }
    },
  });

  useEffect(() => {
    window.parent?.postMessage({ type: "init" }, "*");
  }, []);

  useEffect(() => {
    function listener(event: MessageEvent) {
      if (event.source === window.parent) {
        const message = event.data;
        if (typeof message === "object" && message != null) {
          switch (message.type) {
            case "update": {
              const value: { config: JsonFormsCore["data"] } = message.value;
              if (Object.keys(value.config).length > 0) {
                initConfigData.current = value.config;
                setConfigData({
                  data: value.config,
                  errors: [],
                  defaults: false,
                });
                break;
              }
            }
          }
        }
      }
    }

    window.addEventListener("message", listener);
    return () => window.removeEventListener("message", listener);
  }, []);

  const isInputResetable = useMemo(() => {
    if (!schemas.input) return false;
    return (
      JSON.stringify(defaults(schemas.input)) !== JSON.stringify(inputData.data)
    );
  }, [schemas.input, inputData.data]);

  function onSubmit() {
    if (
      !stopStream &&
      (!!inputData.errors?.length || !!configData.errors?.length)
    ) {
      return;
    }

    if (stopStream) {
      stopStream();
    } else {
      startStream(inputData.data, configData.data);
    }
  }

  const submitRef = useRef<(() => void) | null>(null);
  submitRef.current = onSubmit;

  useEffect(() => {
    window.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        submitRef.current?.();
      }
    });
  }, []);

  const isSendDisabled =
    !stopStream && (!!inputData.errors?.length || !!configData.errors?.length);

  if (!schemas.config || !schemas.input) {
    return <></>;
  }

  return (
    <AppCallbackContext.Provider value={callbacks}>
      <div className="flex items-center flex-col text-ls-black bg-gradient-to-b from-[#F9FAFB] to-[#EFF8FF] min-h-[100dvh] dark:from-[#0C111C] dark:to-[#0C111C]">
        <div className="flex flex-col flex-grow gap-4 px-4 pt-6 max-w-[800px] w-full">
          <h1 className="text-2xl text-left">
            <strong>🦜 LangServe</strong> Playground
          </h1>

          {Object.keys(schemas.config).length > 0 && (
            <div className="flex flex-col gap-3 [&:has(.content>.vertical-layout:first-child:last-child:empty)]:hidden">
              {!isEmbedded && (
                <h2 className="text-xl font-semibold">Configure</h2>
              )}

              <div className="content flex flex-col gap-3">
                <JsonForms
                  schema={schemas.config}
                  data={configData.data}
                  renderers={renderers}
                  cells={cells}
                  onChange={({ data, errors }) =>
                    data
                      ? setConfigData({ data, errors, defaults: false })
                      : undefined
                  }
                />

                {!!configData.errors?.length && configData.data && (
                  <div className="bg-background rounded-xl">
                    <div className="bg-red-500/10 text-red-700 dark:text-red-300 rounded-xl p-3">
                      <strong className="font-bold">Validation Errors</strong>
                      <ul className="list-disc pl-5">
                        {configData.errors?.map((e, i) => (
                          <li key={i}>{e.message}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {!isEmbedded && (
            <div className="flex flex-col gap-3">
              <h2 className="text-xl font-semibold">Try it</h2>

              <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-2xl bg-background">
                <div className="flex items-center justify-between">
                  <h3 className="font-medium">Inputs</h3>
                  {isInputResetable && (
                    <button
                      type="button"
                      className="text-sm px-1 -mr-1 py-0.5 rounded-md hover:bg-divider-500/50 active:bg-divider-500 text-ls-gray-100"
                      onClick={() =>
                        setInputData({
                          data: defaults(schemas.input),
                          errors: [],
                        })
                      }
                    >
                      Reset
                    </button>
                  )}
                </div>

                <JsonForms
                  schema={schemas.input}
                  data={inputData.data}
                  renderers={renderers}
                  cells={cells}
                  onChange={({ data, errors }) =>
                    setInputData({ data, errors })
                  }
                />
                {!!inputData.errors?.length && inputData.data && (
                  <div className="bg-red-500/10 text-red-700 dark:text-red-300 rounded-xl p-3">
                    <strong className="font-bold">Validation Errors</strong>
                    <ul className="list-disc pl-5">
                      {inputData.errors?.map((e, i) => (
                        <li key={i}>{e.message}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {latest && (
                <div className="flex flex-col gap-3">
                  <h2 className="text-xl font-semibold">Output</h2>
                  <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-2xl bg-background text-lg whitespace-pre-wrap break-words">
                    <StreamOutput streamed={latest.streamed_output} />
                  </div>
                  <IntermediateSteps latest={latest} />
                </div>
              )}
            </div>
          )}

          <div className="flex-grow md:hidden" />

          <div className="gap-4 grid grid-cols-2 sticky -mx-4 px-4 py-4 bottom-0 bg-background md:static md:bg-transparent">
            <div className="md:hidden absolute inset-x-0 bottom-full h-5 bg-gradient-to-t from-black/5 to-black/0" />

            {isEmbedded ? (
              <>
                <button
                  type="button"
                  className="px-4 py-3 gap-3 font-medium border border-divider-700 rounded-full flex items-center justify-center hover:bg-divider-500/50 active:bg-divider-500 transition-colors"
                  onClick={() =>
                    window.parent?.postMessage({ type: "close" }, "*")
                  }
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="px-4 py-3 gap-3 font-medium border border-transparent rounded-full flex items-center justify-center bg-blue-500 hover:bg-blue-600 active:bg-blue-700 disabled:opacity-50 transition-colors"
                  onClick={() => {
                    const hash = compressToEncodedURIComponent(
                      JSON.stringify(configData.data)
                    );

                    const state = getStateFromUrl(window.location.href);
                    const targetUrl = `${state.basePath}/c/${hash}`;
                    window.parent?.postMessage(
                      {
                        type: "apply",
                        value: { targetUrl, config: configData.data },
                      },
                      "*"
                    );
                  }}
                >
                  <span className="text-white">Apply</span>
                </button>
              </>
            ) : (
              <>
                <ShareDialog config={configData.data}>
                  <button
                    type="button"
                    className="px-4 py-3 gap-3 font-medium border border-divider-700 rounded-full flex items-center justify-center hover:bg-divider-500/50 active:bg-divider-500 transition-colors"
                  >
                    <ShareIcon className="flex-shrink-0" /> <span>Share</span>
                  </button>
                </ShareDialog>
                <button
                  type="button"
                  className={cn(
                    "px-4 py-3 gap-3 font-medium border border-transparent rounded-full flex items-center justify-center bg-blue-500 disabled:opacity-50 transition-colors",
                    !isSendDisabled
                      ? "hover:bg-blue-600 active:bg-blue-700"
                      : ""
                  )}
                  onClick={onSubmit}
                  disabled={isSendDisabled}
                >
                  {stopStream ? (
                    <>
                      <div role="status">
                        <svg
                          aria-hidden="true"
                          className="w-5 h-5 animate-spin text-white fill-ls-blue"
                          viewBox="0 0 100 101"
                          fill="none"
                          xmlns="http://www.w3.org/2000/svg"
                        >
                          <path
                            d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z"
                            fill="currentColor"
                          />
                          <path
                            d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z"
                            fill="currentFill"
                          />
                        </svg>
                        <span className="sr-only">Loading...</span>
                      </div>
                      <span className="text-white">Stop</span>
                    </>
                  ) : (
                    <>
                      <SendIcon className="flex-shrink-0" />
                      <span className="text-white">Start</span>
                    </>
                  )}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </AppCallbackContext.Provider>
  );
}

export default App;
