import { P, T } from './kernel';

/**
 * Basic Recursive Descent Parser for Propositions and Tactics
 */
export class Parser {
  private chars: string[];
  private pos: number = 0;

  constructor(input: string) {
    this.chars = input.split('');
  }

  private peek(): string {
    return this.chars[this.pos] || '';
  }

  private next(): string {
    return this.chars[this.pos++] || '';
  }

  private skipWhitespace() {
    while (/\s/.test(this.peek())) this.pos++;
  }

  /**
   * Parse a Proposition
   * Precedence: imp (3) < or (2) < and (1)
   */
  public parseProp(): P {
    return this.parseImp();
  }

  private parseImp(): P {
    let left = this.parseOr();
    this.skipWhitespace();
    if (this.peek() === '→' || (this.peek() === '-' && this.chars[this.pos + 1] === '>')) {
      if (this.peek() === '-') this.pos++; // skip '-'
      this.next(); // skip '→' or '>'
      const right = this.parseImp(); // Right associative
      return P.imp(left, right);
    }
    return left;
  }

  private parseOr(): P {
    let left = this.parseAnd();
    while (true) {
      this.skipWhitespace();
      if (this.peek() === '∨' || (this.peek() === '|' && this.chars[this.pos + 1] === '|')) {
        if (this.peek() === '|') this.pos += 2; else this.next();
        const right = this.parseAnd();
        left = P.or(left, right);
      } else {
        break;
      }
    }
    return left;
  }

  private parseAnd(): P {
    let left = this.parsePrimary();
    while (true) {
      this.skipWhitespace();
      if (this.peek() === '∧' || (this.peek() === '&' && this.chars[this.pos + 1] === '&')) {
        if (this.peek() === '&') this.pos += 2; else this.next();
        const right = this.parsePrimary();
        left = P.and(left, right);
      } else {
        break;
      }
    }
    return left;
  }

  private parsePrimary(): P {
    this.skipWhitespace();
    const char = this.peek();

    if (char === '(') {
      this.next();
      const p = this.parseProp();
      this.skipWhitespace();
      if (this.peek() === ')') this.next();
      return p;
    }

    if (char === '⊥' || char === '!' || (char === 'f' && this.chars.slice(this.pos, this.pos + 4).join('') === 'fals')) {
      if (char === 'f') this.pos += 4; else this.next();
      return P.fals();
    }

    // Variable
    let name = '';
    while (/[a-zA-Z0-9]/.test(this.peek())) {
      name += this.next();
    }
    if (name.length > 0) return P.var(name);

    throw new Error(`Unexpected character at pos ${this.pos}: ${char}`);
  }

  /**
   * Parse a Tactic
   */
  public parseTactic(): T {
    this.skipWhitespace();
    let cmd = '';
    while (/[a-z]/.test(this.peek())) {
      cmd += this.next();
    }

    switch (cmd) {
      case 'intro': return T.intro();
      case 'constructor': return T.constructor();
      case 'left': return T.left();
      case 'right': return T.right();
      case 'exact':
      case 'apply':
      case 'compose':
      case 'cases': {
        this.skipWhitespace();
        let numStr = '';
        while (/[0-9]/.test(this.peek())) numStr += this.next();
        const n = parseInt(numStr, 10);
        if (cmd === 'exact') return T.exact(n);
        if (cmd === 'apply') return T.apply(n);
        if (cmd === 'compose') return T.compose(n);
        return T.cases(n);
      }
      case 'lem': {
        const p = this.parseProp();
        return T.lem(p);
      }
      default:
        throw new Error(`Unknown tactic: ${cmd}`);
    }
  }
}

export function parseP(input: string): P {
  return new Parser(input).parseProp();
}

export function parseT(input: string): T {
  return new Parser(input).parseTactic();
}
