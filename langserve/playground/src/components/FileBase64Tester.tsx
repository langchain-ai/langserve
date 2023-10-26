import { ChangeEvent } from "react";
import { withJsonFormsControlProps } from "@jsonforms/react";
import { rankWith, and, schemaMatches, isControl } from "@jsonforms/core";
import { isJsonSchemaExtra } from "../utils/schema";

export const fileBase64Tester = rankWith(
  12,
  and(
    isControl,
    schemaMatches((schema) => {
      if (!isJsonSchemaExtra(schema)) return false;
      return schema.extra.widget.type === "base64file";
    })
  )
);

export const FileBase64ControlRenderer = withJsonFormsControlProps((props) => {
  const handleFileUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = () => {
      const base64String = reader.result as string | null;
      if (base64String != null) {
        const prefix = base64String.indexOf("base64,") + "base64,".length;
        props.handleChange(props.path, base64String.slice(prefix));
      }
    };

    reader.readAsDataURL(file);
  };

  return (
    <div className="control">
      <label className="text-xs uppercase font-semibold text-ls-gray-100">
        {props.label}
      </label>

      <input type="file" onChange={handleFileUpload} />
    </div>
  );
});
