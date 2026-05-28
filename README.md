# Paper Review Blog

A minimal, fast paper review blog built with [Astro](https://astro.build) + [Tailwind CSS v4](https://tailwindcss.com). Designed for researchers and engineers to publish structured paper summaries with LaTeX formulas, full-text search, and clean reading experience.

**[Live Demo](https://xllm-io.github.io/paper-review/)**

## Features

- **LaTeX formula rendering** — inline and display math via KaTeX
- **Full-text search** — client-side search modal (`Cmd/Ctrl+K`)
- **Auto table of contents** — collapsible sidebar with active heading tracking
- **Custom content repo** — load blog posts from a separate private repository at build time
- **Dual deployment** — GitHub Pages and Cloudflare Pages out of the box
- **Mobile-first** — responsive design with scrollable tables and formula overflow handling
- **Auto frontmatter fix** — Python script to validate and repair blog post metadata

## Quick Start

```bash
npm install
npm run dev          # http://localhost:4321
```

## Build & Deploy

```bash
# Cloudflare Pages (default, root path)
npm run build

# GitHub Pages (sub-path)
BASE=/paper-review/ SITE=https://xllm-io.github.io/ npm run build
```

## Custom Content Repository

Load blog Markdown from a private repository while keeping the site code public.

**Local development:**

```bash
BLOG_CONTENT_DIR=/path/to/your/blog-content npm run dev
```

**GitHub Actions** — configure these in your repo settings:

| Type | Name | Value |
|------|------|-------|
| Variable | `CUSTOM_CONTENT_REPO` | `owner/repo` |
| Secret | `PAT_TOKEN` | PAT with repo read access |

Content is checked out from the repo root. Ensure your content repo has `.md` files with valid frontmatter (see [Content Schema](#content-schema)).

## Content Schema

Each `.md` file in the content directory requires:

```yaml
---
title: "Paper Title"
description: "Brief summary of the paper's contribution."
date: 2025-01-15
tags:
  - Deep Learning
  - Paper Summary
draft: false
---
```

Run the frontmatter checker to auto-fix issues:

```bash
python3 fix-frontmatter.py           # check + fix
python3 fix-frontmatter.py --check   # check only
python3 fix-frontmatter.py --images  # fix missing image refs
```

## Project Structure

```
src/
├── components/
│   ├── blog/          # BlogCard, TagBadge, TableOfContents, SearchModal
│   ├── global/        # Navigation
│   └── landing/       # Hero
├── content/blog/      # Default blog content (override via BLOG_CONTENT_DIR)
├── layouts/           # BaseLayout
├── pages/             # index, blog/[slug], tags/[tag], api/search.json
└── styles/            # global.css (Tailwind + custom styles)
```

## Commands

| Command | Action |
|---------|--------|
| `npm run dev` | Start dev server |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `python3 fix-frontmatter.py` | Validate & fix blog frontmatter |

## License

[GPL-3.0](https://opensource.org/licenses/GPL-3.0)
