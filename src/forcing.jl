struct Forcing{Tu,Tv,Tw,TT,TS}
  u::Tu
  v::Tv
  w::Tw
  T::TT
  S::TS
end

@inline zero_func(u, v, w, T, S, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k) = 0

function Forcing(Tu, Tv, Tw, TT, TS)
    if Tu == nothing
        Tu = zero_func
    end
    if Tv == nothing
        Tv = zero_func
    end
    if Tw == nothing
        Tw = zero_func
    end
    if TT == nothing
        TT = zero_func
    end
    if TS == nothing
        TS = zero_func
    end
    Forcing{typeof(Tu),typeof(Tv),typeof(Tw),typeof(TT),typeof(Tu)}(Tu, Tv, Tw, TT, TS)
end
