"Dummy function and forcing default."
@inline zero_func(args...) = 0

"""
    Forcing(Fu, Fv, Fw, FF, FS)

    Forcing(; Fu=zero_func, Fv=zero_func, Fw=zero_func, FT=zero_func, FS=zero_func)

Construct a `Forcing` to specify functions that force `u`, `v`, `w`, `T`, and `S`.
Forcing functions default to `zero_func`, which does nothing.

Forcing functions have the following function signature:
    f(grid::Grid, u::A, v::A, w::A, T::A, S::A, i::Int, j::Int, k::Int)
where A <: AbstractArray, e.g. Array or CuArray.
"""
struct Forcing{Tu,Tv,Tw,TT,TS}
    u::Tu
    v::Tv
    w::Tw
    T::TT
    S::TS
    function Forcing(Fu, Fv, Fw, FT, FS)
        Fu = Fu === nothing ? zero_func : Fu
        Fv = Fv === nothing ? zero_func : Fv
        Fw = Fw === nothing ? zero_func : Fw
        FT = FT === nothing ? zero_func : FT
        FS = FS === nothing ? zero_func : FS
        new{typeof(Fu),typeof(Fv),typeof(Fw),typeof(FT),typeof(FS)}(Fu, Fv, Fw, FT, FS)
    end
end

Forcing(; Fu=nothing, Fv=nothing, Fw=nothing, FT=nothing, FS=nothing) = Forcing(Fu, Fv, Fw, FT, FS)
