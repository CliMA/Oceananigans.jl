"""
$(TYPEDSIGNATURES)

Return the cube root of `x`, computed without recourse to double precision when
`x isa Float32`.

`Base.cbrt(x::Float32)` performs its Newton refinement in `Float64` (see
`Base._improve_cbrt` in `base/special/cbrt.jl`), so it emits double-precision
instructions even for a `Float32` argument. That is a performance pitfall on most
GPUs, and invalid IR on architectures that have no double precision at all
(e.g. Metal, which additionally provides no native cube-root intrinsic).

The fallback defined here simply calls `Base.cbrt`. Architectures that cannot compile
it override this function in the corresponding package extension.
"""
@inline f32_safe_cbrt(x) = cbrt(x)
