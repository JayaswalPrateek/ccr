<script lang="ts">
  import { currentUser } from '$lib/state';
  import type { UserRole } from '$lib/types';

  export let roles: UserRole[] = [];
</script>

{#if $currentUser && roles.includes($currentUser.role)}
  <slot />
{/if}
