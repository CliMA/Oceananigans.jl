using Oceananigans.Fields: AbstractField

include("conformal_cubed_sphere_grid.jl")

struct ConformalCubedSphereFaceField{X, Y, Z, A, G, B} <: AbstractField{X, Y, Z, A, G}
                   data :: A
                   grid :: G
    boundary_conditions :: B

    function ConformalCubedSphereFaceField{X, Y, Z}(data, grid, bcs) where {X, Y, Z}
        validate_field_data(X, Y, Z, data, grid)
        return new{X, Y, Z, typeof(data), typeof(grid), typeof(bcs)}(data, grid, bcs)
    end
end
