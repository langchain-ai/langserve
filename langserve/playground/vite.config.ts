import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import svgr from "vite-plugin-svgr";

// https://vitejs.dev/config/
export default defineConfig({
  base: "/____LANGSERVE_BASE_URL/",
  plugins: [svgr(), react()],
  server: {
    proxy: {
      "^/____LANGSERVE_BASE_URL.*/(config_schema|input_schema|stream_log|feedback)(/[a-zA-Z0-9-]*)?$": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace("/____LANGSERVE_BASE_URL", ""),
      },
    },
  },
});
