import {
  materialAllOfControlTester,
  MaterialAllOfRenderer,
  MaterialObjectRenderer,
  materialOneOfControlTester,
  MaterialOneOfRenderer,
} from "@jsonforms/material-renderers";
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
import {
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
import {
  customAnyOfTester,
  CustomAnyOfRenderer,
} from "./components/CustomAnyOfRenderer";

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
