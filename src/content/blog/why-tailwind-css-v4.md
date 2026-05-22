---
title: "Why Tailwind CSS v4 Changes Everything"
description: "Tailwind CSS v4 introduces a CSS-first configuration, lightning-fast builds, and a new era for utility-first styling."
date: 2025-03-10
tags: ["tailwind", "css", "web development"]
---

## A New Foundation

Tailwind CSS v4 is a ground-up rewrite. The most significant change: configuration moves from `tailwind.config.js` to CSS itself.

### CSS-First Configuration

Define your theme directly in your CSS file:

```css
@import "tailwindcss";

@theme {
  --font-sans: "Inter", sans-serif;
  --color-primary: oklch(60% 0.2 250);
}
```

No JavaScript config file needed. Your design tokens live where they belong -- in CSS.

## Performance Improvements

The new engine, built in Rust, delivers:

- **10x faster full rebuilds** compared to v3
- **3.78x faster incremental builds**
- **8x faster initial setup**

These are not incremental improvements. They change the development experience fundamentally.

## Key Features

### Native CSS Variables

Theme values become CSS custom properties automatically:

```css
/* These are equivalent in v4 */
color: var(--color-primary);
@apply text-primary;
```

### Container Queries

Built-in container query support:

```html
<div class="@container">
  <p class="@lg:text-lg">Responsive to container</p>
</div>
```

### Improved Variants

New variants for modern CSS features:

- `inverted-colors:` for high contrast mode
- `pointer-fine:` for precise pointers
- `scripting-none:` for no-JS environments

## Migration from v3

Most v3 projects migrate with minimal changes:

1. Replace `tailwind.config.js` with `@theme` in CSS
2. Update any renamed utilities
3. Remove PostCSS config (Tailwind v4 uses Vite directly)

> The best time to adopt v4 is now. The migration is worth it for the DX improvements alone.
