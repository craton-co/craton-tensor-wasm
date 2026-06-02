;; A small 16×16 f32 matrix-multiply fixture for the auto-offload integration
;; test. Three input pointers (A, B, C) and side-length N; the inner loop is
;; a tight FMA over v128 lanes, which the detector should classify as
;; `Offload`.

(module
  (memory (export "mem") 16 16)
  (func $matmul (export "matmul")
        (param $a i32) (param $b i32) (param $c i32) (param $n i32)
    (local $i i32)
    (local $j i32)
    (local $k i32)
    (local $sum f32)
    (local $av f32)
    (local $bv f32)
    (local.set $i (i32.const 0))
    (block $iend
      (loop $iloop
        (br_if $iend (i32.ge_s (local.get $i) (local.get $n)))
        (local.set $j (i32.const 0))
        (block $jend
          (loop $jloop
            (br_if $jend (i32.ge_s (local.get $j) (local.get $n)))
            (local.set $sum (f32.const 0))
            (local.set $k (i32.const 0))
            (block $kend
              (loop $kloop
                (br_if $kend (i32.ge_s (local.get $k) (local.get $n)))
                (local.set $av (f32.load (local.get $a)))
                (local.set $bv (f32.load (local.get $b)))
                (local.set $sum
                  (f32.add (local.get $sum)
                           (f32.mul (local.get $av) (local.get $bv))))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $kloop)))
            (f32.store (local.get $c) (local.get $sum))
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $jloop)))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $iloop)))))
