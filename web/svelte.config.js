import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	preprocess: vitePreprocess(),
	kit: {
		adapter: adapter({
			// Serve index.html for all unmatched routes (SPA mode).
			fallback: '200.html'
		}),
		alias: {
			$lib: 'src/lib',
			$components: 'src/components'
		}
	}
};

export default config;
