import "./App.css";
import { Drawer } from "vaul";

import React, { useEffect, useMemo, useRef, useState } from "react";
import defaults from "json-schema-defaults";
import { JsonForms } from "@jsonforms/react";
import {
  materialAllOfControlTester,
  MaterialAllOfRenderer,
  materialAnyOfControlTester,
  MaterialAnyOfRenderer,
  materialObjectControlTester,
  MaterialObjectRenderer,
  materialOneOfControlTester,
  MaterialOneOfRenderer,
} from "@jsonforms/material-renderers";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import relativeDate from "dayjs/plugin/relativeTime";
import SendIcon from "./assets/SendIcon.svg?react";
import ShareIcon from "./assets/ShareIcon.svg?react";
import {
  compressToEncodedURIComponent,
  decompressFromEncodedURIComponent,
} from "lz-string";

import {
  BooleanCell,
  DateCell,
  DateTimeCell,
  EnumCell,
  IntegerCell,
  NumberCell,
  SliderCell,
  TextAreaCell,
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
} from "@jsonforms/vanilla-renderers";

import { useSchemas } from "./useSchemas";
import { RunState, useStreamLog } from "./useStreamLog";
import { JsonFormsCore } from "@jsonforms/core";

dayjs.extend(relativeDate);
dayjs.extend(utc);

const URL_LENGTH_LIMIT = 2000;

function str(o: unknown): React.ReactNode {
  return typeof o === "object"
    ? JSON.stringify(o, null, 2)
    : (o as React.ReactNode);
}

const renderers = [
  ...vanillaRenderers,

  // use material renderers to handle objects and json schema references
  // they should yield the rendering to simpler cells
  { tester: materialObjectControlTester, renderer: MaterialObjectRenderer },
  { tester: materialAllOfControlTester, renderer: MaterialAllOfRenderer },
  { tester: materialAnyOfControlTester, renderer: MaterialAnyOfRenderer },
  { tester: materialOneOfControlTester, renderer: MaterialOneOfRenderer },
];

const cells = [
  { tester: booleanCellTester, cell: BooleanCell },
  { tester: dateCellTester, cell: DateCell },
  { tester: dateTimeCellTester, cell: DateTimeCell },
  { tester: enumCellTester, cell: EnumCell },
  { tester: integerCellTester, cell: IntegerCell },
  { tester: numberCellTester, cell: NumberCell },
  { tester: sliderCellTester, cell: SliderCell },
  { tester: textAreaCellTester, cell: TextAreaCell },
  { tester: textCellTester, cell: TextAreaCell },
  { tester: timeCellTester, cell: TimeCell },
];

function IntermediateSteps(props: { latest: RunState }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div className="flex flex-col border border-divider-700 rounded-2xl bg-background">
      <button
        className="font-medium text-left p-4"
        onClick={() => setExpanded((open) => !open)}
      >
        Intermediate steps
      </button>
      {expanded && (
        <div className="flex flex-col gap-5 p-4 pt-0 divide-solid divide-y rounded-b-xl">
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

function getStateFromUrl(path: string) {
  let configFromUrl = null;
  let basePath = path;
  if (basePath.endsWith("/")) {
    basePath = basePath.slice(0, -1);
  }

  if (basePath.endsWith("/playground")) {
    basePath = basePath.slice(0, -"/playground".length);
  }

  // check if we can omit the last segment
  const [configHash, c, ...rest] = basePath.split("/").reverse();
  if (c === "c") {
    basePath = rest.reverse().join("/");
    try {
      configFromUrl = JSON.parse(decompressFromEncodedURIComponent(configHash));
    } catch (error) {
      console.error(error);
    }
  }
  return { basePath, configFromUrl };
}

function CopyButton(props: { value: string }) {
  const [copied, setCopied] = useState(false);
  const cbRef = useRef<number | null>(null);

  function toggleCopied() {
    setCopied(true);

    if (cbRef.current != null) window.clearTimeout(cbRef.current);
    cbRef.current = window.setTimeout(() => setCopied(false), 1500);
  }

  useEffect(() => {
    return () => {
      if (cbRef.current != null) {
        window.clearTimeout(cbRef.current);
      }
    };
  }, []);

  return (
    <button
      className="px-2 py-1 border-l border-divider-700"
      onClick={() => {
        navigator.clipboard.writeText(props.value).then(toggleCopied);
      }}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

function ShareDialog(props: { config: unknown }) {
  const hash = useMemo(() => {
    return compressToEncodedURIComponent(JSON.stringify(props.config));
  }, [props.config]);

  const state = getStateFromUrl(window.location.href);

  // get base URL
  const targetUrl = `${state.basePath}/c/${hash}`;

  // .../c/[hash]/playground
  const playgroundUrl = `${targetUrl}/playground`;

  // cURL, JS: .../c/[hash]/invoke
  // Python: .../c/[hash]
  const invokeUrl = `${targetUrl}/invoke`;

  return (
    <div className="bg-background p-4 border border-divider-700 rounded-2xl">
      <h3 className="text-xl font-medium">Share</h3>
      <p>Link to the playground</p>

      <div className="grid grid-cols-[1fr,auto] dark:bg-gray-950 bg-gray-100 border-divider-700 border rounded-md text-xs items-center">
        <div className=" font-mono overflow-auto whitespace-nowrap px-2 no-scrollbar">
          {playgroundUrl}
        </div>
        <CopyButton value={playgroundUrl} />
      </div>

      <p className="text-sm">Copy the code snippet</p>

      <div className="flex flex-col gap-2">
        {targetUrl.length < URL_LENGTH_LIMIT && (
          <>
            <div>Python</div>
            <div className="grid grid-cols-[1fr,auto] dark:bg-gray-950 bg-gray-100 border-divider-700 border rounded-md text-xs items-center">
              <div className=" font-mono overflow-auto whitespace-nowrap px-2 no-scrollbar">
                {targetUrl}
              </div>
              <CopyButton value={targetUrl} />
            </div>
          </>
        )}

        {invokeUrl.length < URL_LENGTH_LIMIT && (
          <>
            <div>cURL (/invoke)</div>
            <div className="grid grid-cols-[1fr,auto] dark:bg-gray-950 bg-gray-100 border-divider-700 border rounded-md text-xs items-center">
              <div className=" font-mono overflow-auto whitespace-nowrap px-2 no-scrollbar">
                {invokeUrl}
              </div>
              <CopyButton value={invokeUrl} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function App() {
  const [isShareOpen, setIsShareOpen] = useState(false);

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
        data: state.configFromUrl ?? defaults(schemas.config),
        errors: [],
      });
      setInputData({ data: defaults(schemas.input), errors: [] });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [schemas.config]);
  // the runner
  const { startStream, stopStream, latest } = useStreamLog();

  return schemas.config && schemas.input ? (
    <div className="flex flex-col gap-4 text-ls-black">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl text-left">
          <strong>
            🦜 <span className="hidden sm:inline">LangServe</span>
          </strong>{" "}
          Playground
        </h1>
        <button
          type="button"
          className="px-4 py-2 gap-2 border border-divider-700 rounded-full flex items-center justify-center hover:bg-divider-500/50 active:bg-divider-500 transition-colors"
          onClick={() => {
            setIsShareOpen((state) => !state);
          }}
        >
          <ShareIcon /> <span>Share</span>
        </button>
      </div>
      <div className="flex flex-col gap-3">
        <h2 className="text-xl font-semibold">Configure</h2>

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
          <>
            <h3>Validation Errors</h3>

            {configData.errors?.map((e, i) => (
              <p key={i}>{e.message}</p>
            ))}
          </>
        )}
      </div>

      <div className="flex flex-col gap-3">
        <h2 className="text-xl font-semibold">Customize</h2>

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
            <>
              <h3>Validation Errors</h3>
              {inputData.errors?.map((e, i) => (
                <p key={i}>{e.message}</p>
              ))}
            </>
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

      {isShareOpen && <ShareDialog config={configData.data} />}

      <div className="flex gap-4 justify-center">
        <button
          type="button"
          className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center border border-transparent disabled:opacity-50"
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
          {stopStream ? <span className="text-white">Stop</span> : <SendIcon />}
        </button>
      </div>

      <div className="text-center text-sm">
        <strong>🦜 LangServe</strong>. An Open Source project by LangChain
      </div>
    </div>
  ) : null;
}

export default App;
