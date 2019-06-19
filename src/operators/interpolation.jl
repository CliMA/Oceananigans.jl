# Operators that interpolate areas.
@inline ϊAx_caa(i, j, k, grid::Grid) where T = T(0.5) * (Ax(i, j, k, grid) + Ax(i+1, j, k, grid))
@inline ϊAy_aca(i, j, k, grid::Grid) where T = T(0.5) * (Ay(i, j, k, grid) + Ay(i, j+1, k, grid))
@inline ϊAz_aac(i, j, k, grid::Grid) where T = T(0.5) * (Az(i, j, k, grid) + Az(i, j, k+1, grid))

@inline ϊAx_faa(i, j, k, grid::Grid) where T = T(0.5) * (Ax(i-1, j, k, grid) + Ax(i, j, k, grid))
@inline ϊAy_afa(i, j, k, grid::Grid) where T = T(0.5) * (Ay(i, j-1, k, grid) + Ay(i, j, k, grid))
@inline ϊAz_aaf(i, j, k, grid::Grid) where T = T(0.5) * (Az(i, j, k-1, grid) + Az(i, j, k, grid))

# Operators that interpolate volumes.
@inline ϊx_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i+1, j, k, grid) + V(i, j, k, grid))
@inline ϊy_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i, j+1, k, grid) + V(i, j, k, grid))
@inline ϊz_V(i, j, k, grid::Grid{T}) where T = T(0.5) * (V(i, j, k+1, grid) + V(i, j, k, grid))

@inline ϊx_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊx_V(i, j, k, grid)
@inline ϊy_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊy_V(i, j, k, grid)
@inline ϊz_V⁻¹(i, j, k, grid::Grid{T}) = 1 / ϊz_V(i, j, k, grid)

# Interpolation operators.
@inline ϊx_caa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i+1, j, k] + f[i,    j, k])
@inline ϊx_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i-1,  j, k])

@inline ϊy_aca(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i, j+1, k] + f[i,    j, k])
@inline ϊy_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (f[i,   j, k] + f[i,  j-1, k])

@inline function ϊz_aac(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k+1] + f[i, j, k])
    end
end

@inline function ϊz_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return T(0.5) * (f[i, j, k] + f[i, j, k-1])
    end
end

# Interpolation operators of the form ϊ(A*f) where A is an area and f is an array.
@inline ϊxAx_caa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i+1, j, k, grid) * f[i+1, j, k] + Ax(i,   j, k, grid) * f[i,    j, k])
@inline ϊxAx_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i,   j, k, grid) * f[i,   j, k] + Ax(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAy_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i,   j, k, grid) * f[i,   j, k] + Ay(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAz_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i-1, j, k, grid) * f[i-1,  j, k])

@inline ϊyAy_aca(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i, j+1, k, grid) * f[i, j+1, k] + Ay(i,   j, k, grid) * f[i,    j, k])
@inline ϊyAx_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ax(i,   j, k, grid) * f[i,   j, k] + Ax(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAy_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Ay(i,   j, k, grid) * f[i,   j, k] + Ay(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAz_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i, j-1, k, grid) * f[i,  j-1, k])

@inline function ϊzAz_aac(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Az(i, j, k+1, grid) * f[i, j, k+1] + Az(i, j, k, grid) * f[i, j, k])
    end
end

@inline function ϊzAx_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Ax(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Ax(i, j, k, grid) * f[i, j, k] + Ax(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAy_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Ay(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Ay(i, j, k, grid) * f[i, j, k] + Ay(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAz_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Az(i, j, k, grid) * f[i, j, k] + Az(i, j, k-1, grid) * f[i, j, k-1])
    end
end
