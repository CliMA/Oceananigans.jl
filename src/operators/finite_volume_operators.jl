using Oceananigans:
    Grid, RegularCartesianGrid, VerticallyStretchedCartesianGrid

@inline Δx(i, j, k, grid::RegularCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δx
@inline Δx(i, j, k, grid::Grid) = @inbounds grid.Δx[i, j, k]

@inline Δy(i, j, k, grid::RegularCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::VerticallyStretchedCartesianGrid) = grid.Δy
@inline Δy(i, j, k, grid::Grid) = @inbounds grid.Δy[i, j, k]

@inline Δz(i, j, k, grid::RegularCartesianGrid) = grid.Δz
@inline Δz(i, j, k, grid::VerticallyStretchedCartesianGrid) = @inbounds grid.Δz[k]
@inline Δz(i, j, k, grid::Grid) = @inbounds grid.Δz[i, j, k]

@inline Ax(i, j, k, grid::Grid) = Δy(i, j, k, grid) * Δz(i, j, k, grid)
@inline Ay(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δz(i, j, k, grid)
@inline Az(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid)

@inline V(i, j, k, grid::Grid) = Δx(i, j, k, grid) * Δy(i, j, k, grid) * Δz(i, j, k, grid)
@inline V⁻¹(i, j, k, grid::Grid) = 1 / V(i, j, k, grid)

@inline δx_c2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i,   j, k] - f[i-1, j, k]
@inline δx_f2c(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_e2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_f2e(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i,   j, k] - f[i-1, j, k]

@inline δy_c2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j,   k] - f[i, j-1, k]
@inline δy_f2c(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_e2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_f2e(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j,   k] - f[i, j-1, k]

@inline function δz_c2f(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

@inline function δz_f2c(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_e2f(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_f2e(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

@inline δxA_c2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i,   j, k, grid) * f[i,   j, k] - Ax(i-1, j, k) * f[i-1, j, k]
@inline δxA_f2c(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i+1, j, k, grid) * f[i+1, j, k] - Ax(i,   j, k) * f[i,   j, k]
@inline δxA_e2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i+1, j, k, grid) * f[i+1, j, k] - Ax(i,   j, k) * f[i,   j, k]
@inline δxA_f2e(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i,   j, k, grid) * f[i,   j, k] - Ax(i-1, j, k) * f[i-1, j, k]

@inline δyA_c2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i,   j, k, grid) * f[i, j,   k] - Ay(i, j-1, k, grid) * f[i, j-1, k]
@inline δyA_f2c(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i, j+1, k, grid) * f[i, j+1, k] - Ay(i, j,   k, grid) * f[i, j,   k]
@inline δyA_e2f(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i, j+1, k, grid) * f[i, j+1, k] - Ay(i, j,   k, grid) * f[i, j,   k]
@inline δyA_f2e(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i,   j, k, grid) * f[i, j,   k] - Ay(i, j-1, k, grid) * f[i, j-1, k]

@inline function δzA_c2f(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return Az(i, j, k-1, grid) * f[i, j, k-1] - Az(i, j, k, grid) * f[i, j, k]
    end
end

@inline function δzA_f2c(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return Az(i, j, k, grid) * f[i, j, k] - Az(i, j, k+1, grid) * f[i, j, k+1]
    end
end

@inline function δzA_e2f(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return Az(i, j, k, grid) * f[i, j, k] - Az(i, j, k+1, grid) * f[i, j, k+1]
    end
end

@inline function δzA_f2e(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return Az(i, j, k-1, grid) * f[i, j, k-1] - Az(i, j, k, grid) * f[i, j, k]
    end
end

@inline ϊx_c2f(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])
@inline ϊx_f2c(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i+1, j, k] + f[i,    j, k])
@inline ϊx_f2e(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])

@inline ϊy_c2f(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])
@inline ϊy_f2c(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i, j+1, k] + f[i,    j, k])
@inline ϊy_f2e(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])

@inline fv(i, j, k, grid::Grid{T}, v::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (avgy_f2c(i-1,  j, k, grid, v) + avgy_f2c(i, j, k, grid, v))
@inline fu(i, j, k, grid::Grid{T}, u::AbstractArray, f::AbstractFloat) where T = T(0.5) * f * (avgx_f2c(i,  j-1, k, grid, u) + avgx_f2c(i, j, k, grid, u))

@inline function ϊz_c2f(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k] + f[i, j, k-1])
    end
end

@inline function ϊz_f2c(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k+1] + f[i, j, k])
    end
end

@inline function ϊz_f2e(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k] + f[i, j, k-1])
    end
end

@inline function div_f2c(i, j, k, grid::Grid, fx::AbstractArray, fy::AbstractArray, fz::AbstractArray)
    V⁻¹(i, j, k, grid) * (δxA_f2c(i, j, k, grid, fx) + δyA_f2c(i, j, k, grid, fy) + δzA_f2c(i, j, k, grid, fz))
end

@inline function div_c2f(i, j, k, grid::Grid, fx::AbstractArray, fy::AbstractArray, fz::AbstractArray)
    V⁻¹(i, j, k, grid) * (δxA_c2f(i, j, k, grid, fx) + δyA_c2f(i, j, k, grid, fy) + δzA_c2f(i, j, k, grid, fz))
end
