import { P, T } from './kernel';

/**
 * Precedence-aware proposition stringification
 */
export function toStringP(p: P, parent?: P, strict: boolean = false): string {
  const getPrecedence = (prop?: P): number => {
    if (!prop) return 999;
    switch (prop.type) {
      case 'fals':
      case 'var': return 0;
      case 'and': return 1;
      case 'or': return 2;
      case 'imp': return 3;
    }
  };

  const thisPrec = getPrecedence(p);
  const parentPrec = getPrecedence(parent);

  const addParens = (s: string): string => {
    if (strict && thisPrec >= parentPrec) return `(${s})`;
    if (!strict && thisPrec > parentPrec) return `(${s})`;
    return s;
  };

  switch (p.type) {
    case 'fals': return '⊥';
    case 'var': return p.name;
    case 'and': return addParens(`${toStringP(p.left, p, true)} ∧ ${toStringP(p.right, p, false)}`);
    case 'or': return addParens(`${toStringP(p.left, p, true)} ∨ ${toStringP(p.right, p, false)}`);
    case 'imp': return addParens(`${toStringP(p.left, p, true)} → ${toStringP(p.right, p, false)}`);
  }
}

/**
 * Tactic stringification
 */
export function toStringT(t: T): string {
  switch (t.type) {
    case 'intro': return 'intro';
    case 'exact': return `exact ${t.n}`;
    case 'apply': return `apply ${t.n}`;
    case 'compose': return `compose ${t.n}`;
    case 'constructor': return 'constructor';
    case 'left': return 'left';
    case 'right': return 'right';
    case 'cases': return `cases ${t.h}`;
    case 'lem': return `lem ${toStringP(t.p)}`;
  }
}
