namespace PropLogicKernel

/--
basic data structures
Prop
Goal  -- a proposition and hypotheses
State --  a list of goals to solve
Tactic -- a rule to change state
--/

-- proposition
inductive P where
  | fals: P
  | var (name: String): P
  | and (this: P) (that: P): P
  | or (this: P) (that: P): P
  | imp (this: P) (that: P): P
  deriving BEq, Ord

-- tactic
inductive T where
  -- PRIMITIVES
  -- if goal is A → B
  -- add hyp h: A and replace goal with B
  | intro: T
  -- if goal is A and h: A
  -- goal is accomplished
  | exact (n: Nat): T
  -- if goal is B and h: A → B
  -- replace goal with A
  | apply (n: Nat): T
  -- if goal is B and h: A1 → B1
  -- split into two goals A1 and (B1 → B)
  | compose (n: Nat): T
  -- if goal is A ∧ B
  -- split into two goals A and B
  | constructor: T
  -- if goal is A ∨ B
  -- replace goal with A
  | left: T
  -- if goal is A ∨ B
  -- replace goal with B
  | right: T

  -- if h: A ∨ B
  -- branch intro (hyp h₁: A) and (hyp h₂: B)
  -- if h: A ∧ B
  -- add hyp (h₁: A) and (h₂: B)
  -- if h: False
  -- ex falso quodlibet (from False, anything follows)
  -- cases doesn't resolve implication
  | cases (h: Nat): T

  -- CLASSICAL LOGIC
  -- law of excluded middle
  -- add hyp ¬ A ∨ A
  | lem (p: P): T

class Ctx (α: Type u) where
  empty : α
  get? (ctx: α) (n: Nat): Option P
  set (ctx: α) (n: Nat) (p: P): α
  iter (ctx: α): List (Nat × P)

-- goal
structure G (α: Type) [Ctx α] where
  hyp: α
  goal: P
  deriving BEq

partial def T.resolveGoal? [Ctx α] (t: T) (vc: Nat) (cl : Bool) (g: G α): Option (Nat × List (G α)) :=
  -- (h: Option Nat) => (h: Option P)
  let h?: Option P :=
    let n?: Option Nat :=
      match t with
        | .exact n => some n
        | .apply n => some n
        | .compose n => some n
        | .cases n => some n
        | _ => none
    match n? with
      | none => none
      | some n => Ctx.get? g.hyp n

  match (g.goal, t, h?) with
    -- GOAL RESOLUTION

    -- if goal is A → B
    -- add hyp h: A and replace goal with B
    | (.imp A B, .intro, _) =>
      some (vc+1, [
        {g with hyp := Ctx.set g.hyp vc A, goal := B},
      ])

    -- if goal is A and h: A
    -- goal is accomplished
    | (A, .exact _, some A1) =>
      if A == A1 then
        some (vc, [])
      else
        none

    -- if goal is B and h: A → B
    -- replace goal with A
    | (B, .apply _, some (.imp A1 B1)) =>
      if B == B1 then
        some (vc, [
          {g with goal := A1},
        ])
      else
        none

    -- if goal is B and h: A1 → B1
    -- split into two goals A1 and (B1 → B)
    | (B, .compose _, some (.imp A1 B1)) =>
      some (vc, [
        {g with goal := A1},
        {g with goal := .imp B1 B},
      ])


    -- GOAL DECOMPOSITION

    -- if goal is A ∧ B
    -- split into two goals A and B
    | (.and A B, .constructor, _) =>
      some (vc, [
        {g with goal := A},
        {g with goal := B},
      ])
    -- if goal is A ∨ B
    -- replace goal with A
    | (.or A B, .left, _) =>
      some (vc, [
        {g with goal := A},
      ])
    -- if goal is A ∨ B
    -- replace goal with B
    | (.or A B, .right, _) =>
      some (vc, [
        {g with goal := B},
      ])

    -- HYPOTHESIS DECOMPOSITION

    -- if h: A ∨ B
    -- branch intro (hyp h₁: A) and (hyp h₂: B)
    -- if h: A ∧ B
    -- add hyp (h₁: A) and (h₂: B)
    -- if h: False
    -- done ex falso quodlibet (from False, anything follows)
    -- cases doesn't resolve implication
    | (_, .cases _, some (.or A B)) =>
      some (vc + 2, [
        {g with hyp := Ctx.set g.hyp vc A},
        {g with hyp := Ctx.set g.hyp (vc+1) B},
      ])
    | (_, .cases _, some (.and A B)) =>
      some (vc + 2, [
        {g with hyp := Ctx.set (Ctx.set g.hyp vc A) (vc+1) B},
      ])
    | (_, .cases _, some (.fals)) =>
      some (vc, [])

    -- CLASSICAL LOGIC

    -- law of excluded middle
    -- add hyp ¬ A ∨ A
    | (_, .lem A, _) =>
      if ¬ cl then none else
      some (vc + 1, [
        {g with hyp := Ctx.set g.hyp vc (P.or (.imp A .fals) A)},
      ])

    | _ => none

-- state
structure S (α: Type) [Ctx α] where
  varCount: Nat   -- handled by tactic
  sorrCount: Nat  -- handled by REPL
  newCount: Nat   -- handled by REPL
  stack: List (G α)

def T.resolveState? [Ctx α] (t: T) (cl: Bool) (s: S α): Option (S α) :=
  match (s.stack) with
    | [] => none
    | g :: gs =>
      match t.resolveGoal? s.varCount cl g with
        | none => none
        | some (vc, ga) => some
          {s with
            varCount := vc,
            stack := ga ++ gs,
          }

end PropLogicKernel
