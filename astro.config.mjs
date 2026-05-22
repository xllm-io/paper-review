import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import sitemap from "@astrojs/sitemap";
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
// https://astro.build/config
// Set BASE and SITE env vars for sub-path deployments, e.g.:
//   GitHub Pages:  BASE=/paper-review/ SITE=https://xllm-io.github.io/ npm run build
//   Cloudflare:    npm run build  (defaults to /)
export default defineConfig({
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [[rehypeKatex, { throwOnError: false, trust: true }]],
    shikiConfig: {
      theme: "one-dark-pro",
    },
  },
  base: process.env.BASE || "/",
  site: process.env.SITE || "https://yourwebsite.com/",
  integrations: [ sitemap()]
});