/**
 * Auth helpers — login, logout, token initialisation.
 */

import { goto } from '$app/navigation';
import { api } from './api';
import { authToken, currentUser } from './state';
import { get } from 'svelte/store';

export async function initAuth(): Promise<boolean> {
  const token = get(authToken);
  if (!token) return false;

  try {
    api.setToken(token);
    // Timeout prevents the layout spinner from hanging forever when the
    // server is unreachable or still starting up.
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('Auth check timed out')), 8000),
    );
    const user = await Promise.race([api.me(), timeout]);
    currentUser.set(user);
    return true;
  } catch {
    logout();
    return false;
  }
}

export async function login(username: string, password: string): Promise<void> {
  const resp = await api.login(username, password);
  authToken.set(resp.access_token);
  api.setToken(resp.access_token);
  const user = await api.me();
  currentUser.set(user);
}

export function logout(): void {
  authToken.set(null);
  currentUser.set(null);
  api.setToken(null);
  goto('/login');
}
