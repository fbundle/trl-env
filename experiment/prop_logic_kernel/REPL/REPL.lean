
namespace REPL

structure Step α where
  state: α
  code: UInt32
  err: List String
  out: List String

def Step.map (s: Step α) (f: α → β): Step β :=
  {
    state := f s.state,
    code := s.code,
    err := s.err,
    out := s.out,
  }

abbrev Transition α := α → String → Step α

partial def run (trans: Transition α) (prev: Step α)
  (errPrefix: String := "-- ")
  (outPrefix: String := "")
  (prompt: String := "> ")
: IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr

  let err := String.intercalate "" (prev.err.map (λ line => errPrefix ++ line ++ "\n"))
  let out := String.intercalate "" (prev.out.map (λ line => outPrefix ++ line ++ "\n"))

  stderr.putStr err
  stdout.putStr out
  stderr.flush
  stdout.flush

  stderr.putStr prompt
  stderr.flush
  let line ← stdin.getLine
  if line.isEmpty then
    -- EOF: no more input
    pure prev.code
  else
    let current := trans prev.state line
    run trans current errPrefix outPrefix prompt

end REPL
