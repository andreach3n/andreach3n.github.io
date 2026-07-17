// @ts-check
import { defineConfig } from 'astro/config';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// https://astro.build/config
export default defineConfig({
	// Your production URL. This lets Astro generate correct absolute links,
	// sitemaps, and canonical URLs for your GitHub Pages site.
	site: 'https://andreach3n.github.io',

	// Render LaTeX written in Markdown ($…$ inline, $$…$$ display) at build
	// time. The KaTeX stylesheet that makes it look right is loaded by
	// BaseLayout, so any page using that layout is covered.
	markdown: {
		remarkPlugins: [remarkMath],
		rehypePlugins: [rehypeKatex],
	},
});
