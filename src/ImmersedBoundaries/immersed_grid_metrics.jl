
const c = Center()
const f = Face()

using Oceananigans.AbstractOperations: GridMetricOperation
import Oceananigans.Grids: solid_node, solid_interface

"""
    `solid_node` returns true only if a location is completely immersed
    `solid_interface` returns true if a location is partially immersed
        
    as an example (in a 1D immersed grid) :

     Immersed      Fluid
    ----------- ...........
   |     ∘     |     ∘ 
   f     c     f     c
  i-1   i-1    i     i

     `solid_interface(f, c, c, i, 1, 1, grid) = true`
     `solid_node(f, c, c, i, 1, 1, grid)      = false`

     `solid_node(Center(), Center(), Center(), args...) == solid_interface(Center(), Center(), Center(), args...)` as 
     `Center(), Center(), Center()` can be only either fully immersed or not at all 

     `solid_interface` is used in `GridMetricOperation` to assess the grid metric value. 
     `metric = ifelse(solid_interface, 0.0, metric)` eliminating values inside the immersed domain and values lying on the
      immersed boundary from the reduction

      `is_immersed_boundary` returns true only if the interface has a solid and a fluid side (the actual immersed boundary)
      which is true only when `solid_node = false` and `solid_interface = true` (as the case of the face at `i` above)
"""

@inline solid_node(i, j, k, ibg::IBG) = is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary) | solid_node(i, j, k, ibg.grid)

# Defining all the metrics for Immersed Boundaries

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ), LZ in (:ᶜ, :ᶠ)
    for dir in (:x, :y, :z), operator in (:Δ, :A)
    
        metric = Symbol(operator, dir, LX, LY, LZ)
        @eval begin
            import Oceananigans.Operators: $metric
            @inline $metric(i, j, k, ibg::IBG) = $metric(i, j, k, ibg.grid)
        end
    end

    volume = Symbol(:V, LX, LY, LZ)
    @eval begin
        import Oceananigans.Operators: $volume
        @inline $volume(i, j, k, ibg::IBG) = $volume(i, j, k, ibg.grid)
    end
end

@inline Δzᵃᵃᶜ(i, j, k, ibg::IBG) = Δzᵃᵃᶜ(i, j, k, ibg.grid)
@inline Δzᵃᵃᶠ(i, j, k, ibg::IBG) = Δzᵃᵃᶠ(i, j, k, ibg.grid)
