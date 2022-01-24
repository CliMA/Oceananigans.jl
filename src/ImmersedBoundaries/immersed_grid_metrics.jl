
const c = Center()
const f = Face()

using Oceananigans.AbstractOperations: GridMetricOperation

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

# fallback for not-immersed grid
@inline solid_node(i, j, k, grid)     = false
@inline solid_node(i, j, k, ibg::IBG) = is_immersed(i, j, k, ibg.grid, ibg.immersed_boundary)

@inline solid_node(LX, LY, LZ, i, j, k, ibg)      = solid_node(i, j, k, ibg)
@inline solid_interface(LX, LY, LZ, i, j, k, ibg) = solid_node(i, j, k, ibg)

@inline solid_node(::Face, LY, LZ, i, j, k, ibg) = solid_node(i, j, k, ibg) & solid_node(i-1, j, k, ibg)
@inline solid_node(LX, ::Face, LZ, i, j, k, ibg) = solid_node(i, j, k, ibg) & solid_node(i, j-1, k, ibg)
@inline solid_node(LX, LY, ::Face, i, j, k, ibg) = solid_node(i, j, k, ibg) & solid_node(i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, LZ, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) & solid_node(c, f, c, i-1, j, k, ibg)
@inline solid_node(::Face, LY, ::Face, i, j, k, ibg) = solid_node(c, c, f, i, j, k, ibg) & solid_node(c, c, f, i-1, j, k, ibg)
@inline solid_node(LX, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, c, i, j, k, ibg) & solid_node(c, f, c, i, j, k-1, ibg)

@inline solid_node(::Face, ::Face, ::Face, i, j, k, ibg) = solid_node(c, f, f, i, j, k, ibg) & solid_node(c, f, f, i-1, j, k, ibg)

@inline solid_interface(::Face, LY, LZ, i, j, k, ibg) = solid_node(i, j, k, ibg) | solid_node(i-1, j, k, ibg)
@inline solid_interface(LX, ::Face, LZ, i, j, k, ibg) = solid_node(i, j, k, ibg) | solid_node(i, j-1, k, ibg)
@inline solid_interface(LX, LY, ::Face, i, j, k, ibg) = solid_node(i, j, k, ibg) | solid_node(i, j, k-1, ibg)

@inline solid_interface(::Face, ::Face, LZ, i, j, k, ibg) = solid_interface(c, f, c, i, j, k, ibg) | solid_interface(c, f, c, i-1, j, k, ibg)
@inline solid_interface(::Face, LY, ::Face, i, j, k, ibg) = solid_interface(c, c, f, i, j, k, ibg) | solid_interface(c, c, f, i-1, j, k, ibg)
@inline solid_interface(LX, ::Face, ::Face, i, j, k, ibg) = solid_interface(c, f, c, i, j, k, ibg) | solid_interface(c, f, c, i, j, k-1, ibg)

@inline solid_interface(::Face, ::Face, ::Face, i, j, k, ibg) = solid_interface(c, f, f, i, j, k, ibg) | solid_interface(c, f, f, i-1, j, k, ibg)

@inline is_immersed_boundary(LX, LY, LZ, i, j, k, ibg) = solid_interface(LX, LY, LZ, i, j, k, ibg) & !solid_node(LX, LY, LZ, i, j, k, ibg)

for metric in (
               :Δxᶜᶜᵃ,
               :Δxᶜᶠᵃ,
               :Δxᶠᶠᵃ,
               :Δxᶠᶜᵃ,

               :Δyᶜᶜᵃ,
               :Δyᶜᶠᵃ,
               :Δyᶠᶠᵃ,
               :Δyᶠᶜᵃ,

               :Δzᵃᵃᶜ,
               :Δzᵃᵃᶠ,
               :Δzᶠᶜᶜ,
               :Δzᶜᶠᶜ,
               :Δzᶠᶜᶠ,
               :Δzᶜᶠᶠ,

               :Azᶜᶜᵃ,
               :Azᶜᶠᵃ,
               :Azᶠᶠᵃ,
               :Azᶠᶜᵃ,

               :Axᶜᶜᶜ, 
               :Axᶠᶜᶜ,
               :Axᶠᶠᶜ,
               :Axᶜᶠᶜ,
               :Axᶠᶜᶠ,
               :Axᶜᶜᶠ,
               
               :Ayᶜᶜᶜ,
               :Ayᶜᶠᶜ,
               :Ayᶠᶜᶜ,
               :Ayᶠᶠᶜ,
               :Ayᶜᶠᶠ,
               :Ayᶜᶜᶠ,

               :Vᶜᶜᶜ, 
               :Vᶠᶜᶜ,
               :Vᶜᶠᶜ,
               :Vᶜᶜᶠ,
              )

    @eval begin
        import Oceananigans.Operators: $metric
        @inline $metric(i, j, k, ibg::ImmersedBoundaryGrid) = $metric(i, j, k, ibg.grid)
    end
end