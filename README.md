# Astro & Tailwind CSS Starter Kit



## License

This template is open-source software licensed under the [GPL-3.0 license](https://opensource.org/licenses/GPL-3.0). Feel free to fork, modify, and use it in your projects.


# This template is using Tailwind CSS V4

Now we are using only a CSS file. It's called `global.css` and it's located in the src/styles folder. Now we are eimporting Tailwind CSS on the same file instead of using the `tailwind.config.cjs` file. Like this:

```css
// Importing Tailwind CSS
@import "tailwindcss";
// Importing Tailwind plugins
@plugin "@tailwindcss/typography";
@plugin "@tailwindcss/forms";
```

Then to add your styles you will use the @theme directive. Like this:

```css
@theme {
  /* Your CSS goes here, see how styles are written on the global.css file */
}
```

Remember this is just in Alpha version, so you can use it as you want. Just keep an eye on the changes that Tailwind CSS is going to make.
## Template Structure

Inside of your Astro project, you'll see the following folders and files:

```
/
├── public/
├── src/
│   └── pages/
│       └── index.astro
└── package.json
```

Astro looks for `.astro` or `.md` files in the `src/pages/` directory. Each page is exposed as a route based on its file name.

There's nothing special about `src/components/`, but that's where we like to put any Astro/React/Vue/Svelte/Preact components.

Any static assets, like images, can be placed in the `public/` directory.

## Commands

All commands are run from the root of the project, from a terminal:

| Command                | Action                                           |
| :--------------------- | :----------------------------------------------- |
| `npm install`          | Installs dependencies                            |
| `npm run dev`          | Starts local dev server at `localhost:3000`      |
| `npm run build`        | Build your production site to `./dist/`          |
| `npm run preview`      | Preview your build locally, before deploying     |
| `npm run astro ...`    | Run CLI commands like `astro add`, `astro check` |
| `npm run astro --help` | Get help using the Astro CLI                     |

## Custom content repository

This project supports loading blog Markdown from a private repository at build time while keeping the site rendering logic unchanged (`getCollection("blog")`).

### Local development

- Default behavior (no env var): loads from `./src/content/blog`
- Override content directory:

```bash
BLOG_CONTENT_DIR=/absolute/path/to/custom-blog-content npm run build
```

`BLOG_CONTENT_DIR` must point to a directory containing the blog markdown files (and related assets such as `figures/` if used).

### CI / GitHub Actions

The Pages workflow can optionally pull blog content from a private repo:

- Repository variable `BLOG_CONTENT_REPO` (format: `owner/repo`)
- Optional repository variable `CUSTOM_CONTENT_REPO_REF` (branch/tag/SHA)
- Repository secret `CUSTOM_CONTENT_REPO_TOKEN` (PAT or deploy token with read access to the private content repo)

If these are configured, CI checks out the private repo and sets `BLOG_CONTENT_DIR` for Astro build.  
If not configured, CI falls back to `./src/content/blog`.

> Note: private source markdown can be kept non-public, but generated website pages are still publicly accessible after deployment.

## Want to learn more?

Feel free to check Lexington's [documentation](https://lexingtonthemes.com/documentation/quick-start/)
