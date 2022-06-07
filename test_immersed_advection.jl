
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

grid = RectilinearGrid(size=(10, 10), extent=(10, 10), topology=(Bounded, Bounded, Flat))
ibg  = ImmersedBoundaryGrid(grid, GridFittedBoundary((x, y, z) -> (x < 5 || y < 5)))

c = CenterField(ibg)
u = XFaceField(ibg)
v = YFaceField(ibg)
set!(c, 1.0)
set!(u, 1.0)
set!(v, 1.0)

using Oceananigans.ImmersedBoundaries: mask_immersed_field!

wait(mask_immersed_field!(c))
wait(mask_immersed_field!(u))
wait(mask_immersed_field!(v))

using Oceananigans.BoundaryConditions

fill_halo_regions!((u, v, c))

using Test
using Oceananigans.Advection: 
        _left_biased_interpolate_xᶜᵃᵃ, 
        _left_biased_interpolate_xᶠᵃᵃ, 
        _right_biased_interpolate_xᶜᵃᵃ,
        _right_biased_interpolate_xᶠᵃᵃ,
        _left_biased_interpolate_yᵃᶜᵃ, 
        _left_biased_interpolate_yᵃᶠᵃ, 
        _right_biased_interpolate_yᵃᶜᵃ,
        _right_biased_interpolate_yᵃᶠᵃ

for adv in [UpwindBiasedFifthOrder(), UpwindBiasedThirdOrder(), WENO5()]
    for i in 6:9, j in 6:9
        @show i, j, adv
        @show @test _left_biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, adv, c) ≈ 1.0
        @show @test _right_biased_interpolate_xᶠᵃᵃ(i+1, j, 1, ibg, adv, c) ≈ 1.0
        @show @test _left_biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, adv, c) ≈ 1.0
        @show @test _right_biased_interpolate_yᵃᶠᵃ(i, j+1, 1, ibg, adv, c) ≈ 1.0
    end
end