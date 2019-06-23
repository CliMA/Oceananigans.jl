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
@inline ϊxAxF_caa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AxF(i+1, j, k, grid) * f[i+1, j, k] + AxF(i,   j, k, grid) * f[i,    j, k])
@inline ϊxAxF_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AxF(i,   j, k, grid) * f[i,   j, k] + AxF(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAyF_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AyF(i,   j, k, grid) * f[i,   j, k] + AyF(i-1, j, k, grid) * f[i-1,  j, k])
@inline ϊxAz_faa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i-1, j, k, grid) * f[i-1,  j, k])

@inline ϊyAyF_aca(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AyF(i, j+1, k, grid) * f[i, j+1, k] + AyF(i,   j, k, grid) * f[i,    j, k])
@inline ϊyAxF_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AxF(i,   j, k, grid) * f[i,   j, k] + AxF(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAyF_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (AyF(i,   j, k, grid) * f[i,   j, k] + AyF(i, j-1, k, grid) * f[i,  j-1, k])
@inline ϊyAz_afa(i, j, k, grid::Grid{T}, f::AbstractArray) where T = @inbounds T(0.5) * (Az(i,   j, k, grid) * f[i,   j, k] + Az(i, j-1, k, grid) * f[i,  j-1, k])

@inline function ϊzAzF_aac(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == grid.Nz
        @inbounds return T(0.5) * AzF(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (AzF(i, j, k+1, grid) * f[i, j, k+1] + AzF(i, j, k, grid) * f[i, j, k])
    end
end

@inline function ϊzAxF_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return AxF(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (AxF(i, j, k, grid) * f[i, j, k] + AxF(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAyF_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return AyF(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (AyF(i, j, k, grid) * f[i, j, k] + AyF(i, j, k-1, grid) * f[i, j, k-1])
    end
end

@inline function ϊzAz_aaf(i, j, k, grid::Grid{T}, f::AbstractArray) where T
    if k == 1
        @inbounds return Az(i, j, k, grid) * f[i, j, k]
    else
        @inbounds return T(0.5) * (Az(i, j, k, grid) * f[i, j, k] + Az(i, j, k-1, grid) * f[i, j, k-1])
    end
end
