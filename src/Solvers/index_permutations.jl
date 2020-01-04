@inline permute_index(::PressureSolver{CPU}, i, j, k, Nx, Ny, Nz) = i, j, k

@inline function _permute_index(i, N)
    if (i & 1) == 1  # Same as isodd(i)
        return convert(UInt32, CUDAnative.floor(i/2) + 1)
    else
        return convert(UInt32, N - CUDAnative.floor((i-1)/2))
    end
end

@inline function _unpermute_index(i, N)
    if i <= N/2
        return 2i-1
    else
        return 2(N-i+1)
    end
end

const HPPS = HorizontallyPeriodicPressureSolver
const CPS  = ChannelPressureSolver

@inline   permute_index(::HPPS{GPU}, i, j, k, Nx, Ny, Nz) = i, j,   _permute_index(k, Nz)
@inline unpermute_index(::HPPS{GPU}, i, j, k, Nx, Ny, Nz) = i, j, _unpermute_index(k, Nz)

@inline   permute_index(::CPS{GPU}, i, j, k, Nx, Ny, Nz) = i,   _permute_index(j, Ny),   _permute_index(k, Nz)
@inline unpermute_index(::CPS{GPU}, i, j, k, Nx, Ny, Nz) = i, _unpermute_index(j, Ny), _unpermute_index(k, Nz)
