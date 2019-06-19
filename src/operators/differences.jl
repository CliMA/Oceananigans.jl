# Difference operators.
@inline δx_caa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i+1, j, k] - f[i,   j, k]
@inline δx_faa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i,   j, k] - f[i-1, j, k]

@inline δy_aca(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j+1, k] - f[i, j,   k]
@inline δy_afa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds f[i, j,   k] - f[i, j-1, k]

@inline function δz_aac(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_aaf(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

# Difference operators of the form δ(A*f) where A is an area and f is an array.
@inline δxA_caa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i+1, j, k, grid) * f[i+1, j, k] - Ax(i,   j, k) * f[i,   j, k]
@inline δxA_faa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ax(i,   j, k, grid) * f[i,   j, k] - Ax(i-1, j, k) * f[i-1, j, k]

@inline δyA_aca(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i, j+1, k, grid) * f[i, j+1, k] - Ay(i, j,   k, grid) * f[i, j,   k]
@inline δyA_afa(i, j, k, grid::Grid, f::AbstractArray) = @inbounds Ay(i,   j, k, grid) * f[i, j,   k] - Ay(i, j-1, k, grid) * f[i, j-1, k]

@inline function δzA_aac(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return Az(i, j, k, grid) * f[i, j, k] - Az(i, j, k+1, grid) * f[i, j, k+1]
    end
end

@inline function δzA_aaf(i, j, k, g::Grid{T}, f::AbstractArray) where T
    if k == 1
        return -zero(T)
    else
        @inbounds return Az(i, j, k-1, grid) * f[i, j, k-1] - Az(i, j, k, grid) * f[i, j, k]
    end
end
