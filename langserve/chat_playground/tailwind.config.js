/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        popover: {
          DEFAULT: "hsl(var(--popover))",
        },
        background: {
          DEFAULT: "var(--background)",
        },
        button: {
          "green": "var(--button-green)",
          "green-disabled": "var(--button-green-disabled)",
          "inline": "var(--button-inline)"
        },
        ls: {
          blue: "hsl(211.5, 91.8%, 61.8%)",
          black: "hsl(var(--ls-black))",
          gray: {
            100: "hsl(var(--ls-gray-100))",
            200: "hsl(var(--ls-gray-200))",
            300: "hsl(var(--ls-gray-300))",
            400: "hsl(var(--ls-gray-400))",
          },
        },
        divider: {
          500: "hsl(var(--divider-500))",
          700: "hsl(var(--divider-700))",
        },
      },
    },
  },
  plugins: [],
};
