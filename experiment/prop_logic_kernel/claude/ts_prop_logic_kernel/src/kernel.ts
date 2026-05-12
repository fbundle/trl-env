export type P =
  | { type: 'fals' }
  | { type: 'var'; name: string }
  | { type: 'and'; left: P; right: P }
  | { type: 'or'; left: P; right: P }
  | { type: 'imp'; left: P; right: P };

export const P = {
  fals: (): P => ({ type: 'fals' }),
  var: (name: string): P => ({ type: 'var', name }),
  and: (left: P, right: P): P => ({ type: 'and', left, right }),
  or: (left: P, right: P): P => ({ type: 'or', left, right }),
  imp: (left: P, right: P): P => ({ type: 'imp', left, right }),
};

export function equalsP(a: P, b: P): boolean {
  switch (a.type) {
    case 'fals': return b.type === 'fals';
    case 'var': return b.type === 'var' && a.name === b.name;
    case 'and': return b.type === 'and' && equalsP(a.left, b.left) && equalsP(a.right, b.right);
    case 'or':  return b.type === 'or'  && equalsP(a.left, b.left) && equalsP(a.right, b.right);
    case 'imp': return b.type === 'imp' && equalsP(a.left, b.left) && equalsP(a.right, b.right);
  }
}

export type T =
  | { type: 'intro' }
  | { type: 'exact'; n: number }
  | { type: 'apply'; n: number }
  | { type: 'compose'; n: number }
  | { type: 'constructor' }
  | { type: 'left' }
  | { type: 'right' }
  | { type: 'cases'; h: number }
  | { type: 'lem'; p: P };

export const T = {
  intro: (): T => ({ type: 'intro' }),
  exact: (n: number): T => ({ type: 'exact', n }),
  apply: (n: number): T => ({ type: 'apply', n }),
  compose: (n: number): T => ({ type: 'compose', n }),
  constructor: (): T => ({ type: 'constructor' }),
  left: (): T => ({ type: 'left' }),
  right: (): T => ({ type: 'right' }),
  cases: (h: number): T => ({ type: 'cases', h }),
  lem: (p: P): T => ({ type: 'lem', p }),
};

export type Ctx = Map<number, P>;

export interface G {
  hyp: Ctx;
  goal: P;
}

export function equalsG(a: G, b: G): boolean {
  if (!equalsP(a.goal, b.goal)) return false;
  if (a.hyp.size !== b.hyp.size) return false;
  for (const [k, v] of a.hyp) {
    const bv = b.hyp.get(k);
    if (!bv || !equalsP(v, bv)) return false;
  }
  return true;
}

export interface S {
  varCount: number;
  sorryCount: number;
  newCount: number;
  stack: G[];
}

export function resolveGoal(t: T, vc: number, cl: boolean, g: G): { vc: number; goals: G[] } | null {
  const getHyp = (n: number): P | undefined => g.hyp.get(n);

  switch (t.type) {
    case 'intro':
      if (g.goal.type === 'imp') {
        const newHyp = new Map(g.hyp);
        newHyp.set(vc, g.goal.left);
        return { vc: vc + 1, goals: [{ hyp: newHyp, goal: g.goal.right }] };
      }
      break;

    case 'exact': {
      const h = getHyp(t.n);
      if (h && equalsP(g.goal, h)) {
        return { vc, goals: [] };
      }
      break;
    }

    case 'apply': {
      const h = getHyp(t.n);
      if (h && h.type === 'imp' && equalsP(g.goal, h.right)) {
        return { vc, goals: [{ hyp: g.hyp, goal: h.left }] };
      }
      break;
    }

    case 'compose': {
      const h = getHyp(t.n);
      if (h && h.type === 'imp') {
        return {
          vc,
          goals: [
            { hyp: g.hyp, goal: h.left },
            { hyp: g.hyp, goal: { type: 'imp', left: h.right, right: g.goal } },
          ],
        };
      }
      break;
    }

    case 'constructor':
      if (g.goal.type === 'and') {
        return {
          vc,
          goals: [
            { hyp: g.hyp, goal: g.goal.left },
            { hyp: g.hyp, goal: g.goal.right },
          ],
        };
      }
      break;

    case 'left':
      if (g.goal.type === 'or') {
        return { vc, goals: [{ hyp: g.hyp, goal: g.goal.left }] };
      }
      break;

    case 'right':
      if (g.goal.type === 'or') {
        return { vc, goals: [{ hyp: g.hyp, goal: g.goal.right }] };
      }
      break;

    case 'cases': {
      const h = getHyp(t.h);
      if (!h) break;
      if (h.type === 'or') {
        const hyp1 = new Map(g.hyp);
        hyp1.set(vc, h.left);
        const hyp2 = new Map(g.hyp);
        hyp2.set(vc + 1, h.right);
        return {
          vc: vc + 2,
          goals: [
            { hyp: hyp1, goal: g.goal },
            { hyp: hyp2, goal: g.goal },
          ],
        };
      }
      if (h.type === 'and') {
        const newHyp = new Map(g.hyp);
        newHyp.set(vc, h.left);
        newHyp.set(vc + 1, h.right);
        return { vc: vc + 2, goals: [{ hyp: newHyp, goal: g.goal }] };
      }
      if (h.type === 'fals') {
        return { vc, goals: [] };
      }
      break;
    }

    case 'lem':
      if (cl) {
        const newHyp = new Map(g.hyp);
        newHyp.set(vc, { type: 'or', left: { type: 'imp', left: t.p, right: { type: 'fals' } }, right: t.p });
        return { vc: vc + 1, goals: [{ hyp: newHyp, goal: g.goal }] };
      }
      break;
  }

  return null;
}

export function resolveState(t: T, cl: boolean, s: S): S | null {
  if (s.stack.length === 0) return null;
  const [g, ...gs] = s.stack;
  const res = resolveGoal(t, s.varCount, cl, g);
  if (!res) return null;
  return {
    ...s,
    varCount: res.vc,
    stack: [...res.goals, ...gs],
  };
}
