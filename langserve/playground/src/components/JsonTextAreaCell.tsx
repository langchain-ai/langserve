// From https://github.com/eclipsesource/jsonforms/blob/master/packages/vanilla-renderers/src/cells/TextAreaCell.tsx

/*
  The MIT License

  Copyright (c) 2017-2019 EclipseSource Munich
  https://github.com/eclipsesource/jsonforms

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/
import {
  CellProps,
  isMultiLineControl,
  RankedTester,
  rankWith,
} from "@jsonforms/core";
import { withJsonFormsCellProps } from "@jsonforms/react";
import {
  withVanillaCellProps,
  type VanillaRendererProps,
} from "@jsonforms/vanilla-renderers";
import merge from "lodash/merge";
import { AutosizeTextarea } from "./AutosizeTextarea";
import { cn } from "../utils/cn";

function tryJsonParse(str: string) {
  try {
    return JSON.parse(str);
  } catch {
    return str;
  }
}

function tryJsonStringify(obj: unknown): string | unknown {
  try {
    return JSON.stringify(obj);
  } catch {
    return obj;
  }
}

export const TextAreaCell = (props: CellProps & VanillaRendererProps) => {
  const { data, className, id, enabled, config, uischema, path, handleChange } =
    props;
  const appliedUiSchemaOptions = merge({}, config, uischema.options);
  return (
    <AutosizeTextarea
      value={typeof data === "object" ? tryJsonStringify(data) : data ?? ""}
      onChange={(value) =>
        handleChange(path, value === "" ? undefined : tryJsonParse(value))
      }
      className={cn("w-full text-lg", className)}
      id={id}
      disabled={!enabled}
      autoFocus={appliedUiSchemaOptions.focus}
      placeholder={appliedUiSchemaOptions.placeholder}
    />
  );
};

/**
 * Tester for a multi-line string control.
 * @type {RankedTester}
 */
// eslint-disable-next-line react-refresh/only-export-components
export const textAreaCellTester: RankedTester = rankWith(2, isMultiLineControl);

// eslint-disable-next-line react-refresh/only-export-components
export default withJsonFormsCellProps(withVanillaCellProps(TextAreaCell));
