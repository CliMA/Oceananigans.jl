#####
##### cos α and sin α of the angle α between the x and y coordinate lines (α = 90° on orthogonal grids)
#####

for LX in (:ᶜ, :ᶠ), LY in (:ᶜ, :ᶠ)
    cosα = Symbol(:cosα, LX, LY, :ᵃ)
    sinα = Symbol(:sinα, LX, LY, :ᵃ)

    @eval begin
        @inline $cosα(i, j, k, grid::AbstractGrid) = zero(grid)
        @inline $sinα(i, j, k, grid::AbstractGrid) = one(grid)

        @inline $cosα(i, j, k, grid::SG) = @inbounds grid.$cosα[i, j]
        @inline $sinα(i, j, k, grid::SG) = @inbounds grid.$sinα[i, j]
    end
end
