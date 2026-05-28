import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const blogContentDir = process.env.BLOG_CONTENT_DIR || "./src/content/blog";

const blog = defineCollection({
  loader: glob({ pattern: "**/*.md", base: blogContentDir }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string(),
      date: z.coerce.date(),
      updatedDate: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      image: image().optional(),
      draft: z.boolean().default(false),
    }),
});

export const collections = { blog };
