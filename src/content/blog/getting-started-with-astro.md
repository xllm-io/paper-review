---
title: "Getting Started with Astro: A Practical Guide"
description: "Learn how to build fast, content-focused websites with Astro's island architecture and zero-JS-by-default approach."
date: 2025-01-15
tags: ["astro", "web development", "tutorial"]
---

## Why Astro?

Astro is a modern web framework that ships zero JavaScript to the browser by default. It uses an island architecture that lets you hydrate individual components only when needed.

### Key Benefits

- **Performance**: Astro sites load faster because there is no unnecessary JavaScript
- **Simplicity**: Write components in your favorite framework or plain HTML
- **SEO**: Static HTML is inherently search-engine friendly

## Setting Up Your First Project

To create a new Astro project, run:

```bash
npm create astro@latest
```

Choose a template, install dependencies, and start the dev server:

```bash
npm install
npm run dev
```

## Content Collections

Astro's content collections provide type-safe markdown handling. Define your schema in `src/content.config.ts`:

```typescript
import { defineCollection, z } from "astro:content";

const blog = defineCollection({
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
  }),
});
```

### Querying Collections

Use `getCollection` to fetch and filter your content:

```typescript
const posts = await getCollection("blog");
const published = posts.filter((p) => !p.data.draft);
```

## Deployment

Astro supports deployment to Vercel, Netlify, Cloudflare Pages, and more. Build your site with:

```bash
npm run build
```

The output goes to `dist/` by default, ready for any static hosting provider.
