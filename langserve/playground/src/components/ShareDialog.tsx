import { Drawer } from "vaul";
import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import CodeIcon from "../assets/CodeIcon.svg?react";
import PadlockIcon from "../assets/PadlockIcon.svg?react";
import CopyIcon from "../assets/CopyIcon.svg?react";
import CheckCircleIcon from "../assets/CheckCircleIcon.svg?react";
import { compressToEncodedURIComponent } from "lz-string";
import { getStateFromUrl } from "../utils/url";

const URL_LENGTH_LIMIT = 2000;

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
      className="px-3 py-1"
      onClick={() => {
        navigator.clipboard.writeText(props.value).then(toggleCopied);
      }}
    >
      {copied ? <CheckCircleIcon /> : <CopyIcon />}
    </button>
  );
}

export function ShareDialog(props: { config: unknown; children: ReactNode }) {
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

  const pythonSnippet = `
from langserve import RemoteRunnable

chain = RemoteRunnable("${targetUrl}")
chain.invoke({ ... })
`;

  const typescriptSnippet = `
import { RemoteRunnable } from "langchain/runnables/remote";

const chain = new RemoteRunnable({ url: \`${invokeUrl}\` });
const result = await chain.invoke({ ... });
`;

  return (
    <Drawer.Root>
      <Drawer.Trigger asChild>{props.children}</Drawer.Trigger>
      <Drawer.Portal>
        <Drawer.Overlay className="fixed inset-0 bg-black/40" />
        <Drawer.Content className="flex justify-center items-center mt-24 fixed bottom-0 left-0 right-0 text-ls-black !pointer-events-none after:!bg-background">
          <div className="p-4 bg-background max-w-[calc(800px-2rem)] rounded-t-2xl border border-divider-500 border-b-background pointer-events-auto">
            <h3 className="text-xl font-medium">Share</h3>

            <hr className="border-divider-500 my-4 -mx-4" />

            <div className="flex flex-col gap-3">
              {playgroundUrl.length < URL_LENGTH_LIMIT && (
                <div className="flex flex-col gap-2 p-3 rounded-2xl dark:bg-[#2C2C2E] bg-gray-100">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 flex items-center justify-center text-center text-sm bg-background rounded-xl">
                      🦜
                    </div>
                    <span className="font-semibold">Playground</span>
                  </div>
                  <div className="grid grid-cols-[auto,1fr,auto] dark:bg-[#111111] bg-white rounded-xl text-sm items-center">
                    <PadlockIcon className="mx-3" />
                    <div className="overflow-auto whitespace-nowrap py-3 no-scrollbar text-ls-gray-100">
                      {playgroundUrl.split("://")[1]}
                    </div>
                    <CopyButton value={playgroundUrl} />
                  </div>
                </div>
              )}

              <div className="flex flex-col gap-2 p-3 rounded-2xl dark:bg-[#2C2C2E] bg-gray-100">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 flex items-center justify-center text-center text-sm bg-background rounded-xl">
                    <CodeIcon className="w-4 h-4" />
                  </div>
                  <span className="font-semibold">Get the code</span>
                </div>

                {targetUrl.length < URL_LENGTH_LIMIT && (
                  <div className="grid grid-cols-[1fr,auto] dark:bg-[#111111] bg-white rounded-xl text-sm items-center">
                    <div className="overflow-auto whitespace-nowrap px-3 py-3 no-scrollbar text-ls-gray-100">
                      Python SDK
                    </div>
                    <CopyButton value={pythonSnippet.trim()} />
                  </div>
                )}

                {invokeUrl.length < URL_LENGTH_LIMIT && (
                  <div className="grid grid-cols-[1fr,auto] dark:bg-[#111111] bg-white rounded-xl text-sm items-center">
                    <div className="overflow-auto whitespace-nowrap px-3 py-3 no-scrollbar text-ls-gray-100">
                      TypeScript SDK
                    </div>

                    <CopyButton value={typescriptSnippet.trim()} />
                  </div>
                )}
              </div>
            </div>
          </div>
        </Drawer.Content>
      </Drawer.Portal>
    </Drawer.Root>
  );
}
