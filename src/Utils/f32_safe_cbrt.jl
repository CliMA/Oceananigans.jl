# This file works around `Base.cbrt(::Float32)` emitting double-precision instructions,
# and can be deleted in its entirety once that compiles on every backend we support.
# Upstream issue: https://github.com/JuliaGPU/Metal.jl/issues/952

"""
$(TYPEDSIGNATURES)

Return the cube root of `x`, without recourse to double precision when `x isa Float32`.

`Base.cbrt(::Float32)` refines its estimate in `Float64`, which is invalid on
architectures with no double precision (e.g. Metal). Backends that cannot compile
`Base.cbrt` override this function in their package extension.
"""
@inline f32_safe_cbrt(x) = cbrt(x)
