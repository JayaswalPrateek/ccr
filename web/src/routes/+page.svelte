<script lang="ts">
  import { goto } from '$app/navigation';
  import { onMount } from 'svelte';
  import { authToken } from '$lib/state';
  import { get } from 'svelte/store';

  onMount(() => {
    const token = get(authToken);
    goto(token ? '/dashboard' : '/login', { replaceState: true });
  });
</script>
