import { getCollection } from "astro:content";

export async function GET() {
  const posts = (await getCollection("blog"))
    .filter((post) => !post.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf())
    .map((post) => ({
      title: post.data.title,
      description: post.data.description,
      tags: post.data.tags,
      date: post.data.date.toISOString().split("T")[0],
      slug: post.id,
    }));

  return new Response(JSON.stringify(posts), {
    headers: { "Content-Type": "application/json" },
  });
}
