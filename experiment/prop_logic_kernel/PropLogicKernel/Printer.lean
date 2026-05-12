import PropLogicKernel.Kernel

namespace PropLogicKernel.Printer


def toStringProp (p: P) (parent: Option P := none) (strict: Bool := False): String :=
  let precedence (p: Option P): Nat :=
    match p with
      | none => 999
      | some p =>
        match p with
          | .fals => 0
          | .var _ => 0
          | .and _ _ => 1
          | .or _ _ => 2
          | .imp _ _ => 3

  let thisPrec := precedence p
  let parentPrec := precedence parent


  -- right precedence - we prefer to remove parens on the right
  -- A ∧ B ∧ C = A ∧ (B ∧ C)
  -- A ∨ B ∨ C = A ∨ (B ∨ C)
  -- A → B → C = A → (B → C)

  let addOptionalParens (s: String): String :=
    if strict ∧ thisPrec ≥ parentPrec then s!"({s})" else
    if (¬ strict) ∧ thisPrec > parentPrec then s!"({s})" else
    s

  match p with
    | .fals => "⊥"
    | .var name => name
    | .and this that => addOptionalParens s!"{toStringProp this p true} ∧ {toStringProp that p false}"
    | .or this that => addOptionalParens s!"{toStringProp this p true} ∨ {toStringProp that p false}"
    | .imp this that => addOptionalParens s!"{toStringProp this p true} → {toStringProp that p false}"


instance: ToString P where
  toString := toStringProp

#eval (P.imp (P.or (P.and (P.var "P") (P.var "Q")) (P.var "R")) (P.and (P.var "P") (P.or (P.var "Q") (P.var "R"))))
#eval toStringProp (P.imp (P.or (P.and (P.var "P") (P.var "Q")) (P.var "R")) (P.and (P.var "P") (P.or (P.var "Q") (P.var "R"))))

def toStringTactic (t: T): String :=
  match t with
    | .intro => "intro"
    | .exact h => s!"exact {h}"
    | .apply h => s!"apply {h}"
    | .compose h => s!"compose {h}"
    | .constructor => "constructor"
    | .left => "left"
    | .right => "right"
    | .cases h => s!"cases {h}"
    | .lem p => s!"lem {p}"

instance: ToString T where
  toString := toStringTactic

def toStringPropInGoal [Ctx α] (g: G α): String × List (Nat × String) :=
  let goal := s!"{g.goal}"
  let hyp := (Ctx.iter g.hyp).map (λ (n, p) =>
    (n, s!"{p}")
  : Nat × P → Nat × String)

  (goal, hyp)

def toLinesGoal [Ctx α] (g: G α): List String :=
  let (goal, hyp) := toStringPropInGoal g
  let hyp: List String := hyp.map (λ (n, p) => s!"{n}: {p}")
  let goal := s!"⊢ {goal}"
  goal :: hyp

def toStringGoal [Ctx α] (g: G α): String :=
  String.intercalate "\n" (toLinesGoal g).reverse

instance [Ctx α]: ToString (G α)  where
  toString := toStringGoal

end PropLogicKernel.Printer
