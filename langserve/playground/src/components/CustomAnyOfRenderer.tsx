import { JsonFormsDispatch, withJsonFormsAnyOfProps } from "@jsonforms/react";
import {
  rankWith,
  createCombinatorRenderInfos,
  JsonSchema,
  isAnyOfControl,
} from "@jsonforms/core";
import { renderers, cells } from "../renderers";

export const CustomAnyOfRenderer = withJsonFormsAnyOfProps((props) => {
  const anyOfRenderInfos = createCombinatorRenderInfos(
    (props.schema as JsonSchema).anyOf!,
    props.rootSchema,
    "anyOf",
    props.uischema,
    props.path,
    props.uischemas
  );

  // just assume the last type is the selected one
  // for `anyOf` caused by passing inputs from LLMs/Chat Models
  // this will result in showing the Message renderer
  const selectedIndex = anyOfRenderInfos.length - 1;
  const selectedAnyOfRenderInfo = anyOfRenderInfos[selectedIndex];

  return (
    <JsonFormsDispatch
      schema={selectedAnyOfRenderInfo.schema}
      uischema={selectedAnyOfRenderInfo.uischema}
      path={props.path}
      renderers={renderers}
      cells={cells}
    />
  );
});

export const customAnyOfTester = rankWith(3, isAnyOfControl);
