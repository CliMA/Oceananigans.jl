# Delete this file once `Base.cbrt(::Float32)` compiles everywhere: JuliaGPU/Metal.jl#952

"""
$(TYPEDSIGNATURES)

Cube root of `x`, avoiding the `Float64` refinement in `Base.cbrt(::Float32)` that
architectures without double precision cannot compile. Overridden in their extensions.
"""
@inline f32_safe_cbrt(x) = cbrt(x)
