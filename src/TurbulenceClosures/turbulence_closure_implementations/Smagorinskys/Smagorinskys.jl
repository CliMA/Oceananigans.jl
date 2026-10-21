module Smagorinskys

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: @kernel, @index

using Oceananigans: Oceananigans
using Oceananigans.Grids: AbstractGrid, Center
using Oceananigans.Operators: Operators, Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δzᶜᶜᶜ, ℑxyzᶜᶜᶠ,
                              ℑxᶜᶜᶜ, ℑxᶜᶜᶠ, ℑxᶠᶜᶜ, ℑyᶜᶜᶜ, ℑyᶜᶜᶠ, ℑyᶜᶠᶜ, ℑzᶜᶜᶜ, ℑzᶜᶜᶠ,
                              ℑxyᶜᶜᶜ, ℑxyᶜᶠᶠ, ℑxyᶠᶜᶠ, ℑxyᶠᶠᶜ,
                              ℑxzᶜᶜᶜ, ℑxzᶜᶠᶠ, ℑxzᶠᶜᶠ, ℑxzᶠᶠᶜ,
                              ℑyzᶜᶜᶜ, ℑyzᶜᶠᶠ, ℑyzᶠᶜᶠ, ℑyzᶠᶠᶜ
using Oceananigans.Utils: f32_safe_cbrt


import Oceananigans.TurbulenceClosures: buoyancy_force, buoyancy_tracers, step_closure_prognostics!,
                                        initialize_closure_fields!

include("smagorinsky.jl")
include("dynamic_coefficient.jl")
include("lilly_coefficient.jl")
include("scale_invariant_operators.jl")

end
