import js from "@eslint/js";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import globals from "globals";
import tseslint from "typescript-eslint";

const typescriptFiles = ["**/*.{ts,tsx}"];
const testFiles = [
  "**/*.{test,spec}.{ts,tsx}",
  "src/test/**/*.{ts,tsx}",
];

const scopeToTypescript = (config) => ({
  ...config,
  files: typescriptFiles,
});

export default tseslint.config(
  {
    ignores: ["dist/**", "node_modules/**", "src-tauri/**"],
  },
  {
    linterOptions: {
      reportUnusedDisableDirectives: "error",
    },
  },
  {
    ...js.configs.recommended,
    files: ["**/*.{js,mjs,cjs}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: globals.node,
    },
  },
  ...tseslint.configs.strictTypeChecked.map(scopeToTypescript),
  ...tseslint.configs.stylisticTypeChecked.map(scopeToTypescript),
  {
    files: typescriptFiles,
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser,
        ...globals.es2022,
      },
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: {
      "react-hooks": reactHooks,
      "react-refresh": reactRefresh,
    },
    rules: {
      ...reactHooks.configs.recommended.rules,
      ...reactRefresh.configs.vite.rules,
      "@typescript-eslint/consistent-type-imports": [
        "error",
        {
          disallowTypeAnnotations: false,
          fixStyle: "inline-type-imports",
          prefer: "type-imports",
        },
      ],
      "@typescript-eslint/no-deprecated": "error",
      "@typescript-eslint/array-type": [
        "error",
        {
          default: "array-simple",
        },
      ],
      "@typescript-eslint/no-confusing-void-expression": [
        "error",
        {
          ignoreArrowShorthand: true,
          ignoreVoidOperator: false,
        },
      ],
      "@typescript-eslint/no-floating-promises": [
        "error",
        {
          ignoreIIFE: false,
          ignoreVoid: false,
        },
      ],
      "@typescript-eslint/no-misused-promises": "error",
      "@typescript-eslint/no-unnecessary-condition": "error",
      "@typescript-eslint/no-unnecessary-type-assertion": "error",
      "@typescript-eslint/only-throw-error": "error",
      "@typescript-eslint/prefer-readonly": "error",
      "@typescript-eslint/promise-function-async": "error",
      "@typescript-eslint/restrict-template-expressions": [
        "error",
        {
          allowAny: false,
          allowBoolean: true,
          allowNever: false,
          allowNullish: false,
          allowNumber: true,
          allowRegExp: false,
        },
      ],
      "@typescript-eslint/switch-exhaustiveness-check": "error",
      "curly": ["error", "all"],
      "eqeqeq": ["error", "always"],
      "no-console": ["error", { "allow": ["warn", "error"] }],
      "no-implicit-coercion": "error",
    },
  },
  {
    files: ["vite.config.ts"],
    languageOptions: {
      globals: {
        ...globals.node,
      },
    },
  },
  {
    files: testFiles,
    rules: {
      "@typescript-eslint/non-nullable-type-assertion-style": "off",
      "react-refresh/only-export-components": "off",
    },
  },
);
