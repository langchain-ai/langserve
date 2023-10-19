import { JsonSchema } from "@jsonforms/core";
import get from "lodash/get";

// jsonforms doesn't support schemas with root $ref
// so we resolve root $ref and replace it with the actual schema
export function simplifySchema(schema: JsonSchema, root: JsonSchema = schema) {
  if (schema.$ref) {
    const segments = schema.$ref.split("/").filter((s) => !!s && s !== "#");
    return simplifySchema(
      { ...schema, $ref: undefined, ...get(root, segments) },
      root
    );
  } else {
    return schema;
  }
}
