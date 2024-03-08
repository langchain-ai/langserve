import { JsonSchema } from "@jsonforms/core";

type JsonSchemaExtra = JsonSchema & {
  extra: {
    widget: {
      type: string;
      [key: string]: string | number | Array<string | number>;
    };
  };
};

export function isJsonSchemaExtra(x: JsonSchema): x is JsonSchemaExtra {
  if (!("extra" in x && typeof x.extra === "object" && x.extra != null)) {
    return false;
  }

  if (
    !(
      "widget" in x.extra &&
      typeof x.extra.widget === "object" &&
      x.extra.widget != null
    )
  ) {
    return false;
  }

  return true;
}
