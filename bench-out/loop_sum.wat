(module
  (func (export "_start")
    (local $i i32)
    (local $sum i32)
    (local.set $sum (i32.const 0))
    (local.set $i (i32.const 0))
    (block $end
      (loop $loop
        (br_if $end (i32.ge_s (local.get $i) (i32.const 500000000)))
        (local.set $sum (i32.add (local.get $sum) (local.get $i)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop))))
  (func (export "main")
    (local $i i32)
    (local $sum i32)
    (local.set $sum (i32.const 0))
    (local.set $i (i32.const 0))
    (block $end
      (loop $loop
        (br_if $end (i32.ge_s (local.get $i) (i32.const 10000000)))
        (local.set $sum (i32.add (local.get $sum) (local.get $i)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))))
