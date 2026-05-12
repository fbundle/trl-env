import REPL.REPL
import PropLogicKernel.REPL



-- A ∧ B → B ∧ A
-- (A → B) ∧ (B → ⊥) → A → ⊥
-- A → (A → B) → (A → C) → (B ∨ C → D) → D
-- ((P → ⊥) → ⊥) → P -- need classical logic

def main : IO UInt32 :=
  let init := PropLogicKernel.REPL.init
  let repl := PropLogicKernel.REPL.trans

  REPL.run repl init
