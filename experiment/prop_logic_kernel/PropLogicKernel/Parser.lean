import PropLogicKernel.Kernel
import PropLogicKernel.Printer -- for testing

namespace PropLogicKernel.ParserCombinator


def ParseFunc α := List Char → Option (α × List Char)

def ParseFunc.concat (p1: ParseFunc α) (p2: ParseFunc β): ParseFunc (α × β) :=
  λ (xs1: List Char) => do
  let (a, xs2) ← p1 xs1
  let (b, xs3) ← p2 xs2
  return ((a, b), xs3)

infixr:60 " ++ " => ParseFunc.concat

def ParseFunc.orElse (p1: ParseFunc α) (p2: ParseFunc α): ParseFunc α :=
  λ (xs: List Char) => do
  -- p1 xs <|> p2 xs
  match p1 xs with
    | some (a1, xs) => some (a1, xs)
    | none => p2 xs

infixr:50 " || " => ParseFunc.orElse

def ParseFunc.map (p: ParseFunc α) (m: α → β): ParseFunc β :=
  λ (xs: List Char) => do
    let (a, xs) ← p xs
    return (m a, xs)

def parseFail: ParseFunc α := λ _ => none

def parseExact (ch: Char): ParseFunc Char :=
  λ (xs: List Char) =>
    match xs with
      | [] => none
      | x :: rest =>
        if x == ch then
          some (ch, rest)
        else
          none


partial def ParseFunc.repeat (p: ParseFunc α) (xs: List Char): Option (List α × List Char) :=
  let rec loop (ys: Array α) (xs: List Char): Option (List α × List Char) :=
    match p xs with
      | none => some (ys.toList, xs)
      | some (y, xs1) =>
        assert! xs1.length < xs.length -- parser in repeat must consume input
        loop (ys.push y) xs1
  loop #[] xs


end PropLogicKernel.ParserCombinator


-- grammar (not left recursive)

-- Imp  ::= Or ("->" Imp)?        // right
-- Or   ::= And ("∨" Or)?         // right
-- And  ::= Not ("∧" And)?        // right
-- Not  ::= "¬" Not | Atom
-- Atom ::= Var | ⊥ | "(" Imp ")"

namespace PropLogicKernel.Parser

open PropLogicKernel.ParserCombinator

def ParsePropFunc := ParseFunc P

def parseNonEmptyString (chList: List Char) (xs: List Char): Option (String × List Char) := do
  let pList: List (ParseFunc Char) := chList.map parseExact
  -- p1: parse any characters in chList
  let p1: ParseFunc Char := pList.foldr ParseFunc.orElse parseFail
  -- p2:
  let p2: ParseFunc (List Char) := p1.repeat
  let p3: ParseFunc String := p2.map String.ofList

  let (s, rest) ← p3 xs
  if s.length == 0 then
    failure
  else
    return (s, rest)

def parseName: ParseFunc String := parseNonEmptyString "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_".toList

-- Var
def parseVar: ParsePropFunc := parseName.map P.var

-- Fals
def parseFals: ParsePropFunc := (parseExact '⊥').map (λ _ => P.fals)


def makeRightAssocF (f: P → P → P) (left: P) (rightList: List P): P :=
    match rightList with
      | [] => left
      | p :: rest => f left (makeRightAssocF f p rest)


mutual

-- Atom ::= Var | ⊥ | "(" Imp ")"
partial def parseAtom: ParsePropFunc := parseVar || parseFals || ((parseExact '(') ++ parseImp ++ (parseExact ')')).map (λ (_, p, _) => p)

-- Not  ::= "¬" Not | Atom
partial def parseNot: ParsePropFunc := ((parseExact '¬') ++ parseNot).map (λ (_, p) => P.imp p P.fals) || parseAtom


-- B := A (sep B)?
partial def makeRightAssocParseFunc (parseA: ParsePropFunc) (sep: Char) (mk: P → List P → P) (xs: List Char): Option (P × List Char) := do
  let parseB := makeRightAssocParseFunc parseA sep mk

  let (left, xs) ← parseA xs

  let parseSepB := ((parseExact sep) ++ parseB).map (λ (_, p) => p)
  let (rightList, xs) ← parseSepB.repeat xs

  return (mk left rightList, xs)


-- And  ::= Not ("∧" And)?
partial def parseAnd: ParsePropFunc := makeRightAssocParseFunc parseNot '∧' (makeRightAssocF P.and)

-- Or   ::= And ("∨" Or)?
partial def parseOr: ParsePropFunc := makeRightAssocParseFunc parseAnd '∨' (makeRightAssocF P.or)

-- Imp  ::= Or ("→" Imp)?
partial def parseImp: ParsePropFunc := makeRightAssocParseFunc parseOr '→' (makeRightAssocF P.imp)

end


def parseProp? (s: String): Option P := do
  let xs := s.toList.filter (λ x => ¬ x.isWhitespace)
  let (p, _) ← parseImp xs
  return p

#eval parseProp? "A → ⊥"
#eval parseProp? "A ∧ B → B ∧ A"
#eval parseProp? "(A → B) ∧ ¬ B → ¬ A"
#eval parseProp? "A → (A → B) → (A → C) → (B ∨ C → D) → D"
#eval parseProp? "¬¬P → P"


def parsePrefixAndThen (pre: String) (th: String → Option α) (s: String) : Option α :=
  if ¬ s.startsWith pre then none else
  th (s.drop pre.length).toString

def parseTactic? (s: String): Option T :=
  let s := s.trimAscii.toString
  parsePrefixAndThen "intro" (λ _ => some T.intro) s
  <|>
  parsePrefixAndThen "constructor" (λ _ => some T.constructor) s
  <|>
  parsePrefixAndThen "left" (λ _ => some T.left) s
  <|>
  parsePrefixAndThen "right" (λ _ => some T.right) s
  <|>
  parsePrefixAndThen "exact " (λ ss => (String.toNat? ss).map T.exact) s
  <|>
  parsePrefixAndThen "apply " (λ ss => (String.toNat? ss).map T.apply) s
  <|>
  parsePrefixAndThen "compose " (λ ss => (String.toNat? ss).map T.compose) s
  <|>
  parsePrefixAndThen "cases " (λ ss => (String.toNat? ss).map T.cases) s
  <|>
  parsePrefixAndThen "lem " (λ ss => (parseProp? ss).map T.lem) s



#eval parseTactic? "intro"
#eval parseTactic? "constructor"
#eval parseTactic? "apply 0"
#eval parseTactic? "compose 1"
#eval parseTactic? "refine 2"
#eval parseTactic? "cases 3"
#eval parseTactic? "lem A ∨ B"
#eval parseTactic? "new P ∧ Q"
#eval parseTactic? "sorry"

end PropLogicKernel.Parser
