module JULES

using Oceananigans
using Oceananigans: AbstractGrid

using Oceananigans.TurbulenceClosures:
    ConstantIsotropicDiffusivity, DiffusivityFields, with_tracers

export
    Entropy, Energy,
    DryEarth, DryEarth3,
    CompressibleModel,
    time_step!,
    cfl, update_total_density!

include("Operators/Operators.jl")

include("lazy_fields.jl")
include("thermodynamics.jl")
include("pressure_gradients.jl")
include("microphysics.jl")
include("compressible_model.jl")

include("right_hand_sides.jl")
include("time_stepping_kernels.jl")
include("time_stepping.jl")
include("utils.jl")

#####
##### Preserving old behavior
#####

import Oceananigans.BoundaryConditions: NPBC,
    fill_west_halo!, fill_south_halo!, fill_bottom_halo!,
    fill_east_halo!, fill_north_halo!, fill_top_halo!,
    _fill_west_halo!, _fill_south_halo!, _fill_bottom_halo!,
    _fill_east_halo!, _fill_north_halo!, _fill_top_halo! 

  _fill_west_halo!(c, ::NPBC, H, N) = @views @. c.parent[1:1+H, :, :] = 0
 _fill_south_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, 1:1+H, :] = 0
_fill_bottom_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, 1:1+H] = 0

 _fill_east_halo!(c, ::NPBC, H, N) = @views @. c.parent[N+H+1:N+2H, :, :] = 0
_fill_north_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, N+H+1:N+2H, :] = 0
  _fill_top_halo!(c, ::NPBC, H, N) = @views @. c.parent[:, :, N+H+1:N+2H] = 0

sides = (:west, :east, :south, :north, :top, :bottom)
coords = (:x, :x, :y, :y, :z, :z)
  
for (x, side) in zip(coords, sides)
    outername = Symbol(:fill_, side, :_halo!)
    innername = Symbol(:_fill_, side, :_halo!)
    H = Symbol(:H, x)
    N = Symbol(:N, x)
    @eval begin
        $outername(c, bc::NPBC, arch, grid, args...) =
            $innername(c, bc, grid.$(H), grid.$(N))
    end
end

end # module
