import { P, T, G, S, resolveState, equalsG } from './kernel';

/**
 * Get all available tactics for a given goal
 */
export function getAllAvailTactics(g: G): T[] {
  const tactics: T[] = [
    { type: 'intro' },
    { type: 'constructor' },
    { type: 'left' },
    { type: 'right' }
  ];

  // Add tactics with hypothesis parameters
  for (const n of g.hyp.keys()) {
    tactics.push({ type: 'cases', h: n });
    tactics.push({ type: 'exact', n: n });
    tactics.push({ type: 'apply', n: n });
    tactics.push({ type: 'compose', n: n });
  }

  return tactics;
}

/**
 * Depth-First Search for a proof path
 */
function dfs<State, Action>(
  goalState: (s: State) => boolean,
  transition: (s: State, a: Action) => State | null,
  neighbors: (s: State) => Action[],
  state: State
): { state: State; path: Action[] } | null {
  if (goalState(state)) {
    return { state, path: [] };
  }

  const actions = neighbors(state);
  for (const action of actions) {
    const nextState = transition(state, action);
    if (nextState) {
      const result = dfs(goalState, transition, neighbors, nextState);
      if (result) {
        return {
          state: result.state,
          path: [action, ...result.path]
        };
      }
    }
  }

  return null;
}

/**
 * Solve a proof state with a maximum depth
 */
export function solveWithDepth(maxDepth: number, s: S): { state: S; path: T[] } | null {
  const goalState = (item: { s: S; ts: T[] }) => item.s.stack.length === 0;

  const transition = (curr: { s: S; ts: T[] }, t: T): { s: S; ts: T[] } | null => {
    if (curr.ts.length >= maxDepth) return null;

    const s2 = resolveState(t, false, curr.s);
    if (!s2) return null;

    // Prevent immediate cycles
    if (curr.s.stack.length === s2.stack.length) {
      if (equalsG(curr.s.stack[0], s2.stack[0])) {
        return null;
      }
    }

    return { s: s2, ts: [...curr.ts, t] };
  };

  const neighbors = (curr: { s: S; ts: T[] }): T[] => {
    if (curr.ts.length >= maxDepth) return [];
    if (curr.s.stack.length === 0) return [];
    return getAllAvailTactics(curr.s.stack[0]);
  };

  const result = dfs(goalState, transition, neighbors, { s, ts: [] });
  return result ? { state: result.state.s, path: result.state.ts } : null;
}

/**
 * Automated solver with Iterative Deepening
 */
export function autoSolve(maxDepth: number, s: S): { state: S; path: T[] } | null {
  for (let depth = 1; depth <= maxDepth; depth++) {
    const res = solveWithDepth(depth, s);
    if (res) return res;
  }
  return null;
}
