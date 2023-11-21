import { JsonForms } from "@jsonforms/react";
import { JsonFormsCore, JsonSchema } from "@jsonforms/core";
import { renderers, cells } from "../renderers";

export type ConfigValue = Pick<JsonFormsCore, "data" | "errors"> & {
  defaults: boolean;
};

export function SectionConfigure(props: {
  config: JsonSchema | undefined;
  value: ConfigValue;
  onChange: (value: ConfigValue) => void;
}) {
  if (props.config == null || Object.keys(props.config).length === 0) {
    return null;
  }

  return (
    <div className="flex flex-col gap-3 [&:has(.content>.vertical-layout:first-child:last-child:empty)]:hidden">
      <h2 className="text-xl font-semibold">Configure</h2>

      <div className="content flex flex-col gap-3">
        <JsonForms
          schema={props.config}
          data={props.value.data}
          renderers={renderers}
          cells={cells}
          onChange={({ data, errors }) => {
            if (data) {
              props.onChange({ data, errors, defaults: false });
            }
          }}
        />

        {!!props.value.errors?.length && props.value.data && (
          <div className="bg-background rounded-xl">
            <div className="bg-red-500/10 text-red-700 dark:text-red-300 rounded-xl p-3">
              <strong className="font-bold">Validation Errors</strong>
              <ul className="list-disc pl-5">
                {props.value.errors?.map((e, i) => (
                  <li key={i}>{e.message}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
