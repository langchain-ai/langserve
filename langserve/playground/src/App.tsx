import "./App.css";
import React, { useEffect, useState } from "react";
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
import SendIcon from "./assets/SendIcon.svg";

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
    <div className="flex flex-col border border-divider-700 rounded-xl">
      <button
        className="text-xs font-semibold text-left uppercase p-4"
        onClick={() => setExpanded((open) => !open)}
      >
        Intermediate steps
      </button>
      {expanded && (
        <div className="flex flex-col gap-4 p-4 bg-divider-500/50 rounded-b-xl border-t border-divider-700">
          {Object.values(props.latest.logs).map((log) => (
            <div
              className="p-4 gap-3 flex-col min-w-0 flex border border-divider-700 rounded-xl bg-background"
              key={log.id}
            >
              <div className="flex items-center justify-between">
                <strong className="font-medium">{log.name}</strong>
                <p>{dayjs.utc(log.start_time).fromNow()}</p>
              </div>
              <pre className="break-words whitespace-pre-wrap min-w-0 text-sm">
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
    <div className="flex flex-col gap-4 text-ls-black">
      <h1 className="text-2xl font-medium">Playground</h1>
      <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-xl bg-background">
        <h2 className="text-xl font-medium">Configure</h2>

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

      <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-xl bg-background">
        <h2 className="text-xl font-medium">Inputs</h2>

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

      <div className="p-4 border border-divider-700 flex flex-col gap-3 rounded-xl bg-background">
        <h2 className="text-xl font-medium">Output</h2>
        {latest?.streamed_output.map(str).join("") ?? "..."}

        {latest && <IntermediateSteps latest={latest} />}
      </div>

      <div className="flex gap-4 justify-center">
        <button
          className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center"
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
            <img src={SendIcon} alt="Start" className="text-white" />
          )}
        </button>
      </div>
    </div>
  ) : null;
}

export default App;
