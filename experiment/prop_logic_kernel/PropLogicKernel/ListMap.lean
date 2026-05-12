import PropLogicKernel.Kernel

namespace PropLogicKernel.ListMap



structure ListMap α β where
  data: List (α × β)
  deriving BEq

partial def ListMap.get? [BEq α] (map: ListMap α β) (key: α): Option β :=
  match map.data with
    | [] => none
    | (k, v) :: xs =>
      if k == key then
        some v
      else
        get? {data := xs} key

instance: Ctx (ListMap Nat P) where
  empty := {data := []}
  set := λ (map: ListMap Nat P) (key: Nat) (val: P) => {data := (key, val) :: map.data}
  iter := λ (map: ListMap Nat P) => map.data
  get? := ListMap.get?


end PropLogicKernel.ListMap
