import PropLogicKernel.Kernel
import PropLogicKernel.Parser
import PropLogicKernel.Printer
import PropLogicKernel.ListMap
import PropLogicKernel.Auto


import REPL.REPL

namespace PropLogicKernel.REPL

open PropLogicKernel
open PropLogicKernel.ListMap

abbrev State := S (ListMap Nat P)

def MAX_HIST_SIZE: Nat := 20

structure Hist where
  head: State
  tail: List State

def Hist.push (h: Hist) (s: State): Hist :=
  let tail := if h.tail.length >= MAX_HIST_SIZE then
    h.tail.dropLast
  else
    h.tail

  {head := s, tail := h.head :: tail}

def Hist.pop (h: Hist): Hist :=
  match h.tail with
    | [] => h
    | th :: tt => {head := th, tail := tt}

def Hist.make (s: State): Hist :=
  {head := s, tail := []}

def toStringTs (ts: List T): String :=
  let ts := ts.map (λ t => s!"{t}")
  let ts := String.intercalate "][" ts
  s!"[{ts}]"

def getStep (h: Hist) (message?: Option String := none) (hint: Bool := true): REPL.Step Hist :=
  let s := h.head

  let isAllGoalsAccomplished (s: State): Bool :=
    s.stack.isEmpty ∧ (s.sorrCount == 0) ∧ (s.newCount ≥ 1)

  let getCode (s: State): UInt32 :=
    if isAllGoalsAccomplished s then 0 else 1

  let out: List String :=
    match s.stack with
      | [] => []
      | g :: _ =>
        (Printer.toLinesGoal g).reverse

  let status: List String := []

  let status: List String := if s.newCount == 0 then
    status ++ [
      "commands available:",
      "\tnew (p: P)       \t input new goal",
      "\tundo             \t undo last action",
      "\tsorry            \t sorry",
      "\tauto (d: Nat)    \t search",
      "tactics available:",
      "\tintro            \t ",
      "\texact (h: Nat)   \t ",
      "\tapply (h: Nat)   \t ",
      "\tcompose (h: Nat) \t ",
      "\tconstructor      \t ",
      "\tleft             \t ",
      "\tright            \t ",
      "\tcases (h: Nat)   \t ",
      "\tlem (p: P)       \t ",
    ]
  else
    status
  let status: List String :=
    status ++ [s!"new_count {s.newCount} sorry_count {s.sorrCount} var_count {s.varCount} goals_remaining {s.stack.length}"]

  let status: List String :=
    if isAllGoalsAccomplished s then
      status ++ ["all goals accomplished!"]
    else if hint then
      match s.stack with
      | [] => status
      | g :: _ =>
        let ts := Auto.getAllAvailTactics g (checkAhead := true)
        status ++ [s!"hint: {toStringTs ts}"]
    else
      status

  let status: List String :=
    match message? with
      | none => status
      | some message => message :: status

  {
    state := h,
    code := getCode s,
    err := status,
    out := out,
  }

def init: REPL.Step Hist :=
  getStep (Hist.make {
      varCount := 0,
      sorrCount := 0,
      newCount := 0,
      stack := []
    })


def handleEmpty? (hist: Hist) (inputLine: String): Option (REPL.Step Hist) :=
  if inputLine.length == 0 then
    getStep hist
  else
    none

def handleUndo? (hist: Hist) (inputLine: String): Option (REPL.Step Hist) :=
  if inputLine == "undo" then
    getStep hist.pop
  else
    none

def handleSorry? (hist: Hist) (inputLine: String) : Option (REPL.Step Hist) :=
  let state := hist.head
  if inputLine == "sorry" then
    getStep (hist.push {state with
      sorrCount := state.sorrCount + 1,
      stack := state.stack.drop 1,
    })
  else
    none

def handleAuto? (hist: Hist) (inputLine: String): Option (REPL.Step Hist) :=
  match Parser.parsePrefixAndThen "auto " String.toNat? inputLine with
    | some depth =>
      match Auto.autoSolveWithMaxDepth? depth hist.head with
        | some (newS, path) => getStep (hist.push newS) s!"solved: {toStringTs path.reverse}"
        | none => getStep hist s!"unsolvable with depth {depth}"
    | _ => none

def handleNew? (hist: Hist) (inputLine: String) : Option (REPL.Step Hist) :=
  let state := hist.head
  match Parser.parsePrefixAndThen "new " Parser.parseProp? inputLine with
    | some p =>
    getStep (hist.push {state with
      newCount := state.newCount + 1,
      stack := {hyp := Ctx.empty , goal := p} :: state.stack,
    })
    | _ => none

def handleTactic? (hist: Hist) (inputLine: String) (cl: Bool := true) : REPL.Step Hist :=
  let state := hist.head
  match Parser.parseTactic? inputLine with
    | none => getStep hist "parse error"
    | some tactic =>
        match tactic.resolveState? cl state with
          | none => getStep hist "resolve error"
          | some newState => getStep (hist.push newState)



def trans (hist: Hist) (inputLine: String): REPL.Step Hist :=
  let inputLine := inputLine.trimAscii.toString

  let newState? :=
  handleEmpty? hist inputLine
  <|>
  handleUndo? hist inputLine
  <|>
  handleSorry? hist inputLine
  <|>
  handleAuto? hist inputLine
  <|>
  handleNew? hist inputLine


  match newState? with
    | some newState => newState
    | none => handleTactic? hist inputLine



end PropLogicKernel.REPL
