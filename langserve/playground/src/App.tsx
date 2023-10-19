import "./App.css";

import React, { useEffect, useRef, useState } from "react";
import defaults from "json-schema-defaults";
import { JsonForms } from "@jsonforms/react";
import {
  materialAllOfControlTester,
  MaterialAllOfRenderer,
  materialAnyOfControlTester,
  MaterialAnyOfRenderer,
  MaterialObjectRenderer,
  materialOneOfControlTester,
  MaterialOneOfRenderer,
} from "@jsonforms/material-renderers";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import relativeDate from "dayjs/plugin/relativeTime";
import SendIcon from "./assets/SendIcon.svg?react";
import ShareIcon from "./assets/ShareIcon.svg?react";
import ChevronRight from "./assets/ChevronRight.svg?react";
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
import { RunState, useStreamLog } from "./useStreamLog";
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
import { cn } from "./utils/cn";
import { getStateFromUrl, ShareDialog } from "./components/ShareDialog";

dayjs.extend(relativeDate);
dayjs.extend(utc);

function str(o: unknown): React.ReactNode {
  return typeof o === "object"
    ? JSON.stringify(o, null, 2)
    : (o as React.ReactNode);
}

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

const renderers = [
  ...vanillaRenderers,

  // use material renderers to handle objects and json schema references
  // they should yield the rendering to simpler cells
  { tester: isObjectWithPropertiesControl, renderer: MaterialObjectRenderer },
  { tester: materialAllOfControlTester, renderer: MaterialAllOfRenderer },
  { tester: materialAnyOfControlTester, renderer: MaterialAnyOfRenderer },
  { tester: materialOneOfControlTester, renderer: MaterialOneOfRenderer },

  // custom renderers
  { tester: materialArrayControlTester, renderer: CustomArrayControlRenderer },
  { tester: isObject, renderer: InputControl },
];

const nestedArrayControlTester: RankedTester = rankWith(1, (_, jsonSchema) => {
  return jsonSchema.type === "array";
});

const cells = [
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

function IntermediateSteps(props: { latest: RunState }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="flex flex-col border border-divider-700 rounded-2xl bg-background">
      <button
        className="font-medium text-left p-4 flex items-center justify-between"
        onClick={() => setExpanded((open) => !open)}
      >
        <span>Intermediate steps</span>
        <ChevronRight
          className={cn("transition-all", expanded && "rotate-90")}
        />
      </button>
      {expanded && (
        <div className="flex flex-col gap-5 p-4 pt-0 divide-solid divide-y divide-divider-700 rounded-b-xl">
          {Object.values(props.latest.logs).map((log) => (
            <div
              className="gap-3 flex-col min-w-0 flex bg-background pt-3 first-of-type:pt-0"
              key={log.id}
            >
              <div className="flex items-center justify-between">
                <strong className="text-sm font-medium">{log.name}</strong>
                <p className="text-sm">{dayjs.utc(log.start_time).fromNow()}</p>
              </div>
              <pre className="break-words whitespace-pre-wrap min-w-0 text-sm bg-ls-gray-400 rounded-lg p-3">
                {str(log.final_output) ?? "..."}
              </pre>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function App() {
  const [isIframe] = useState(() => window.self !== window.top);

  // it is possible that defaults are being applied _after_
  // the initial update message has been sent from the parent window
  // so we store the initial config data in a ref
  const initConfigData = useRef<JsonFormsCore["data"]>(null);

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
      const state = getStateFromUrl(window.location.href);
      setConfigData({
        data:
          state.configFromUrl ??
          initConfigData.current ??
          defaults(schemas.config),
        errors: [],
      });
      setInputData({ data: {}, errors: [] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemas.config]);
  // the runner
  const { startStream, stopStream, latest } = useStreamLog();

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
                setConfigData({ data: value.config, errors: [] });
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

  return schemas.config && schemas.input ? (
    <div className="flex items-center flex-col text-ls-black bg-gradient-to-b from-[#F9FAFB] to-[#EFF8FF] min-h-[100dvh] dark:from-[#0C111C] dark:to-[#0C111C]">
      <div className="flex flex-col flex-grow gap-4 px-4 pt-6 max-w-[800px] w-full">
        <h1 className="text-2xl text-left">
          <strong>🦜 LangServe</strong> Playground
        </h1>
        <div className="flex flex-col gap-3">
          {!isIframe && <h2 className="text-xl font-semibold">Configure</h2>}

          <JsonForms
            schema={schemas.config}
            data={configData.data}
            renderers={renderers}
            cells={cells}
            onChange={({ data, errors }) =>
              data ? setConfigData({ data, errors }) : undefined
            }
          />
          {!!configData.errors?.length && (
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

        {!isIframe && (
          <div className="flex flex-col gap-3">
            <h2 className="text-xl font-semibold">Try it</h2>

            <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-2xl bg-background">
              <h3 className="font-medium">Inputs</h3>

              <JsonForms
                schema={schemas.input}
                data={inputData.data}
                renderers={renderers}
                cells={cells}
                onChange={({ data, errors }) => setInputData({ data, errors })}
              />
              {!!inputData.errors?.length && (
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
                <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-2xl bg-background text-lg">
                  {latest.streamed_output.map(str).join("") || "..."}
                </div>
                <IntermediateSteps latest={latest} />
              </div>
            )}
          </div>
        )}

        <div className="flex-grow md:hidden" />

        <div className="gap-4 grid grid-cols-2 sticky -mx-4 px-4 py-4 bottom-0 bg-background md:static md:bg-transparent">
          <div className="md:hidden absolute inset-x-0 bottom-full h-5 bg-gradient-to-t from-black/5 to-black/0" />

          {isIframe ? (
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
                className="px-4 py-3 gap-3 font-medium border border-transparent rounded-full flex items-center justify-center bg-blue-500 hover:bg-blue-600 active:bg-blue-700 disabled:opacity-50 transition-colors"
                onClick={() => {
                  stopStream
                    ? stopStream()
                    : startStream(inputData.data, configData.data);
                }}
                disabled={
                  !stopStream &&
                  (!!inputData.errors?.length || !!configData.errors?.length)
                }
              >
                {stopStream ? (
                  <span className="text-white">Stop</span>
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
  ) : null;
}

export default App;
