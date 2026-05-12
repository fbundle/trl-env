import PropLogicKernel.Kernel
import PropLogicKernel.ListMap
import PropLogicKernel.Printer

namespace PropLogicKernel.Auto
open PropLogicKernel.ListMap

def getAllAtomsOfProp (p: P) (nameSet: List String): List String :=
  match p with
    | .var name => nameSet.insert name -- insert without duplicate
    | .and this that => getAllAtomsOfProp this (getAllAtomsOfProp that nameSet)
    | .or this that => getAllAtomsOfProp this (getAllAtomsOfProp that nameSet)
    | .imp this that => getAllAtomsOfProp this (getAllAtomsOfProp that nameSet)
    | _ => nameSet

def getAllAtomsOfGoal [Ctx α] (g: G α) : List String :=
  let nameSet := getAllAtomsOfProp g.goal []
  let rec getAllAtomsOfHyp (hyp: List (Nat × P)) (nameSet: List String): List String :=
    match hyp with
      | [] => nameSet
      | (_, p) :: rest =>
        let nameSet := getAllAtomsOfProp p nameSet
        getAllAtomsOfHyp rest nameSet

  let nameSet := getAllAtomsOfHyp (Ctx.iter g.hyp) nameSet
  nameSet

def eraseAdjacentDups [BEq α] (l: List α): List α :=
  match l with
    | [] => l
    | _ :: [] => l
    | x1 :: x2 :: xs =>
      if x1 == x2 then
        eraseAdjacentDups (x2 :: xs)
      else
        x1 :: eraseAdjacentDups (x2 :: xs)

def CanonicalGoal := G (ListMap Nat P) deriving BEq

-- make goal into a string with unique hypotheses
def canonicalizeGoal [Ctx α] (g: G α): CanonicalGoal :=
  let hypList: List P := (Ctx.iter g.hyp).map (λ (_, p) => p: Nat × P → P)
  let hypList := hypList.mergeSort (λ a b => (compare a b).isLE)
  let hypList := eraseAdjacentDups hypList

  let hypList: List (Nat × P) := List.zip (List.range hypList.length) hypList

  {
    hyp := {data := hypList},
    goal := g.goal,
  }

def equalGoals [Ctx α] (g1: G α) (g2: G α): Bool :=
  (canonicalizeGoal g1) == (canonicalizeGoal g2)

def cartesian (xs : List α) (ys : List β) (prod: Array (α × β) := #[]) : List (α × β) :=
  match xs with
    | [] => prod.toList
    | x :: rest =>
      cartesian rest ys (prod := prod ++ ys.map (λ y => (x, y)))


#eval cartesian [1, 2, 3] ["a", "b", "c"]

def getAllAvailTactics [Ctx α] (g: G α) (checkAhead: Bool): List T :=
  -- tactic without params
  let tArray: Array T := #[T.intro, T.constructor, T.left, T.right]

  -- tactic with params
  let nList: List Nat := (Ctx.iter g.hyp).map (λ (n, _) => n)
  let mList: List (Nat → T) := [T.cases, T.exact, T.apply, T.compose]
  let tArray: Array T := tArray ++ (cartesian mList nList).map (λ (method, n) => method n)

  let tList: List T := tArray.toList

  if ¬ checkAhead then tList else
  let rec loop (tList: List T) (headTListAcc: Array T) (tListAcc: Array T): List T :=
    -- headTListAcc are those that can resolve goal immediately
    -- higher priority accumulation
    match tList with
      | [] => (headTListAcc ++ tListAcc).toList
      | t :: rest =>
        match t.resolveGoal? 0 false g with
          | none => loop rest headTListAcc tListAcc -- cannot resolve, loop
          | some (_, g2 :: []) =>
            if equalGoals g2 g then -- prevent one step loop
              loop rest headTListAcc tListAcc
            else
              loop rest headTListAcc (tListAcc.push t)  -- resolve ok
          | some (_, []) =>
              loop rest (headTListAcc.push t) tListAcc -- resolve in one step
          | _ => loop rest headTListAcc (tListAcc.push t)  -- resolve ok


  loop tList #[] #[]

partial def dfs
  (goalState: α → Bool)
  (transitionFunc: α → β → Option α)
  (neighbourFunc: α → List β)
  (state: α)
: Option α :=
  if goalState state then some state else


  let rec loop (actions: List β): Option α :=
    match actions with
      | [] => none
      | action :: rest =>
        match transitionFunc state action with
          | none => loop rest -- try other branches
          | some nextState =>
            match dfs
              goalState
              transitionFunc
              neighbourFunc
              nextState
            with
              | none => loop rest -- try other branches
              | some goal => goal

  loop (neighbourFunc state)

def solveWithDepth? [Ctx α] (maxDepth: Nat) (s: S α): Option (S α ×List T) := do
  dfs (α := S α × List T) (β := T)
    (goalState := λ ((s, _): S α × List T) => s.stack.length == 0)
    (transitionFunc := λ (s, ts) t => do
      -- no transition at maxDepth
      if ts.length >= maxDepth then failure else

      let s2 ← t.resolveState? (cl := False) s
      -- prevent one step loop
      if s.stack.length == s2.stack.length then
        let g1 ← s.stack.head?
        let g2 ← s2.stack.head?
        if equalGoals g1 g2 then
          failure
        else
          return (s2, t :: ts)
      else
        return (s2, t :: ts)
    )
    (neighbourFunc := λ (s, ts) =>
      -- no neighbourhood at maxDepth
      if ts.length >= maxDepth then [] else
      match s.stack with
        | [] => []
        | g :: _ => getAllAvailTactics g (checkAhead := False)
    )
    (s, [])

partial def autoSolveWithMaxDepth? [Ctx α] (maxDepth: Nat) (s: S α): Option (S α × List T) :=
  -- iterative deepening depth-first search
  let rec loop (depth: Nat): Option (S α × List T) :=
    if depth > maxDepth then none else
    match solveWithDepth? depth s with
      | none => loop (depth + 1)
      | some (newS, path) => some (newS, path)

  loop 1




end PropLogicKernel.Auto
