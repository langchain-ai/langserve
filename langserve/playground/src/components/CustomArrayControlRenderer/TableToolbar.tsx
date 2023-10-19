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
import React from "react";
import {
  ControlElement,
  createDefaultValue,
  JsonSchema,
  ArrayTranslations,
} from "@jsonforms/core";
import { IconButton, TableRow, Tooltip } from "@mui/material";
import PlusIcon from "../../assets/PlusIcon.svg?react";
import ValidationIcon from "./ValidationIcon";
import NoBorderTableCell from "./NoBorderTableCell";

export interface MaterialTableToolbarProps {
  numColumns: number;
  errors: string;
  label: string;
  path: string;
  uischema: ControlElement;
  schema: JsonSchema;
  rootSchema: JsonSchema;
  enabled: boolean;
  translations: ArrayTranslations;
  addItem(path: string, value: any): () => void;
}

const fixedCellSmall = {
  paddingLeft: 0,
  paddingRight: 0,
};

const TableToolbar = React.memo(function TableToolbar({
  numColumns,
  errors,
  label,
  path,
  addItem,
  schema,
  enabled,
  translations,
}: MaterialTableToolbarProps) {
  return (
    <TableRow>
      <NoBorderTableCell colSpan={numColumns} sx={{ verticalAlign: "top" }}>
        <div className="flex items-center gap-2">
          {label && (
            <span className="text-xs uppercase font-semibold text-ls-gray-100">
              {label}
            </span>
          )}
          {errors.length !== 0 && (
            <ValidationIcon id="tooltip-validation" errorMessages={errors} />
          )}
        </div>
      </NoBorderTableCell>
      {enabled ? (
        <NoBorderTableCell align="right" style={fixedCellSmall}>
          <Tooltip
            id="tooltip-add"
            title={translations.addTooltip}
            placement="bottom"
          >
            <IconButton
              aria-label={translations.addAriaLabel}
              onClick={addItem(path, createDefaultValue(schema))}
              size="large"
              sx={{ p: 1 }}
            >
              <PlusIcon className="text-ls-black" />
            </IconButton>
          </Tooltip>
        </NoBorderTableCell>
      ) : null}
    </TableRow>
  );
});

export default TableToolbar;
