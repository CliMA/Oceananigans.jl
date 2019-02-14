using Oceananigans:
    RegularCartesianGrid,
    Field, CellField, FaceField, FaceFieldX, FaceFieldY, FaceFieldZ, EdgeField,
    VelocityFields, TracerFields, PressureFields, SourceTerms, ForcingFields,
    OperatorTemporaryFields

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

# Difference operators.
@inline δx_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k]
@inline δx_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k]
@inline δx_e2f(g::RegularCartesianGrid, f::EdgeField, i, j, k) = @inbounds f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k]
@inline δx_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k]

@inline δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])
@inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
@inline δx!(g::RegularCartesianGrid, f::EdgeField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
@inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::EdgeField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])

@inline δy_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k]
@inline δy_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k]
@inline δy_e2f(g::RegularCartesianGrid, f::EdgeField, i, j, k) = @inbounds f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k]
@inline δy_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k]

@inline δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])
@inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
@inline δy!(g::RegularCartesianGrid, f::EdgeField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
@inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::EdgeField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])

@inline function δz_c2f(g::RegularCartesianGrid, f::CellField, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
end

@inline function δz_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == g.Nz
        @inbounds return f.data[i, j, g.Nz]
    else
        @inbounds return f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz_e2f(g::RegularCartesianGrid, f::EdgeField, i, j, k)
    if k == g.Nz
        @inbounds return f.data[i, j, g.Nz]
    else
        @inbounds return f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f.data[i, j, k-1] - f.data[i, j, k]
    end
end

@inline function δz_e2f(g::RegularCartesianGrid, f::CellField, δzf::FaceField, i, j, k)
    if k == 1
        @inbounds δzf.data[i, j, k] = 0
    else
        @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField, i, j, k)
    if k == g.Nz
        @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
    else
        @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::EdgeField, δzf::FaceField, i, j, k)
    if k == g.Nz
        @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
    else
        @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
    end
end

@inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::EdgeField, i, j, k)
    if k == 1
        @inbounds δzf.data[i, j, k] = 0
    else
        @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
end

@inline avgx_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k])
@inline avgx_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[incmod1(i, g.Nx), j, k] + f.data[i, j, k])
@inline avgx_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k])

@inline avgx!(g::RegularCartesianGrid, f::CellField, favgx::FaceField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]))
@inline avgx!(g::RegularCartesianGrid, f::FaceField, favgx::CellField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[incmod1(i, g.Nx), j, k] + f.data[i, j, k]))
@inline avgx!(g::RegularCartesianGrid, f::FaceField, favgx::EdgeField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]))

@inline avgy_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k])
@inline avgy_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, incmod1(j, g.Ny), k] + f.data[i, j, k])
@inline avgy_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k])

@inline avgy!(g::RegularCartesianGrid, f::CellField, favgy::FaceField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]))
@inline avgy!(g::RegularCartesianGrid, f::FaceField, favgy::CellField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, incmod1(j, g.Ny), k] + f.data[i, j, k]))
@inline avgy!(g::RegularCartesianGrid, f::FaceField, favgy::EdgeField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]))

@inline function avgz_c2f(g::RegularCartesianGrid, f::CellField, i, j, k)
    if k == 1
        @inbounds return f.data[i, j, k]
    else
        @inbounds return  0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
    end
end

@inline function avgz_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == g.Nz
        @inbounds return 0.5f0 * f.data[i, j, k]
    else
        @inbounds return 0.5f0 * (f.data[i, j, incmod1(k, g.Nz)] + f.data[i, j, k])
    end
end

@inline function avgz_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == 1
        @inbounds return f.data[i, j, k]
    else
        @inbounds return 0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
    end
end

@inline function avgz!(g::RegularCartesianGrid, f::CellField, favgz::FaceField, i, j, k)
    if k == 1
        @inbounds favgz.data[i, j, k] = f.data[i, j, k]
    else
        @inbounds favgz.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
    end
end

@inline function avgz!(g::RegularCartesianGrid, f::FaceField, favgz::CellField, i, j, k)
    if k == g.Nz
        @inbounds favgz.data[i, j, k] = 0.5f0 * f.data[i, j, k]
    else
        @inbounds favgz.data[i, j, k] = 0.5f0 * (f.data[i, j, incmod1(k, g.Nz)] + f.data[i, j, k])
    end
end

@inline function avgz!(g::RegularCartesianGrid, f::FaceField, favgz::EdgeField, i, j, k)
    if k == 1
        @inbounds favgz.data[i, j, k] = f.data[i, j, k]
    else
        @inbounds favgz.data[i, j, k] = 0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
    end
end

@inline function div(g::RegularCartesianGrid, fx::FaceFieldX, fy::FaceFieldY, fz::FaceFieldZ, i, j, k)
    (δx_f2c(g, fx, i, j, k) / g.Δx) + (δy_f2c(g, fy, i, j, k) / g.Δy) + (δz_f2c(g, fz, i, j, k) / g.Δz)
end

@inline function div(g::RegularCartesianGrid, fx::CellField, fy::CellField, fz::CellField, i, j, k)
    (δx_c2f(g, fx, i, j, k) / g.Δx) + (δy_c2f(g, fy, i, j, k) / g.Δy) + (δz_c2f(g, fz, i, j, k) / g.Δz)
end

@inline function δx_f2c_ab̄ˣ(g::RegularCartesianGrid, a::FaceFieldX, b::CellField, i, j, k)
    @inbounds (a.data[incmod1(i, g.Nx), j, k] * avgx_c2f(g, b, incmod1(i, g.Nx), j, k) -
               a.data[i,                j, k] * avgx_c2f(g, b, i,                j, k))
end

@inline function δy_f2c_ab̄ʸ(g::RegularCartesianGrid, a::FaceFieldY, b::CellField, i, j, k)
    @inbounds (a.data[i, incmod1(j, g.Ny), k] * avgy_c2f(g, b, i, incmod1(j, g.Ny), k) -
               a.data[i,                j, k] * avgy_c2f(g, b, i, j,                k))
end

@inline function δz_f2c_ab̄ᶻ(g::RegularCartesianGrid, a::FaceFieldZ, b::CellField, i, j, k)
    if k == g.Nz
        @inbounds return a.data[i, j, k] * avgz_c2f(g, b, i, j, k)
    else
        @inbounds return (a.data[i, j,   k] * avgz_c2f(g, b, i, j,   k) -
                          a.data[i, j, k+1] * avgz_c2f(g, b, i, j, k+1))
    end
end

@inline function div_flux(g::RegularCartesianGrid, u::FaceFieldX, v::FaceFieldY, w::FaceFieldZ, Q::CellField, i, j, k)
    if k == 1
        @inbounds return (δx_f2c_ab̄ˣ(g, u, Q, i, j, k) / g.Δx) + (δy_f2c_ab̄ʸ(g, v, Q, i, j, k) / g.Δy) - ((w.data[i, j, 2] * avgz_c2f(g, Q, i, j, 2)) / g.Δz)
    else
        return (δx_f2c_ab̄ˣ(g, u, Q, i, j, k) / g.Δx) + (δy_f2c_ab̄ʸ(g, v, Q, i, j, k) / g.Δy) + (δz_f2c_ab̄ᶻ(g, w, Q, i, j, k) / g.Δz)
    end
end

@inline function δx_c2f_ūˣūˣ(g::RegularCartesianGrid, u::FaceFieldX, i, j, k)
    avgx_f2c(g, u, i, j, k)^2 - avgx_f2c(g, u, decmod1(i, g.Nx), j, k)^2
end

@inline function δy_e2f_v̄ˣūʸ(g::RegularCartesianGrid, u::FaceFieldX, v::FaceFieldY, i, j, k)
    avgx_f2e(g, v, i, incmod1(j, g.Ny), k) * avgy_f2e(g, u, i, incmod1(j, g.Ny), k) -
    avgx_f2e(g, v, i,                j, k) * avgy_f2e(g, u, i,                j, k)
end

@inline function δz_e2f_w̄ˣūᶻ(g::RegularCartesianGrid, u::FaceFieldX, w::FaceFieldZ, i, j, k)
    if k == g.Nz
        @inbounds return avgx_f2e(g, w, i, j, k) * avgz_f2e(g, u, i, j, k)
    else
        @inbounds return avgx_f2e(g, w, i, j,   k) * avgz_f2e(g, u, i, j,   k) -
                         avgx_f2e(g, w, i, j, k+1) * avgz_f2e(g, u, i, j, k+1)
    end
end

@inline function u∇u(g::RegularCartesianGrid, U::VelocityFields, i, j, k)
    (δx_c2f_ūˣūˣ(g, U.u, i, j, k) / g.Δx) + (δy_e2f_v̄ˣūʸ(g, U.u, U.v, i, j, k) / g.Δy) + (δz_e2f_w̄ˣūᶻ(g, U.u, U.w, i, j, k) / g.Δz)
end

@inline function δx_e2f_ūʸv̄ˣ(g::RegularCartesianGrid, u::FaceFieldX, v::FaceFieldY, i, j, k)
    avgy_f2e(g, u, incmod1(i, g.Nx), j, k) * avgx_f2e(g, v, incmod1(i, g.Nx), j, k) -
    avgy_f2e(g, u, i,                j, k) * avgx_f2e(g, v, i,                j, k)
end

@inline function δy_c2f_v̄ʸv̄ʸ(g::RegularCartesianGrid, v::FaceFieldY, i, j, k)
    avgy_f2c(g, v, i, j, k)^2 - avgy_f2c(g, v, i, decmod1(j, g.Ny), k)^2
end

@inline function δz_e2f_w̄ʸv̄ᶻ(g::RegularCartesianGrid, v::FaceFieldY, w::FaceFieldZ, i, j, k)
    if k == g.Nz
        @inbounds return avgy_f2e(g, w, i, j, k) * avgz_f2e(g, v, i, j, k)
    else
        @inbounds return avgy_f2e(g, w, i, j,   k) * avgz_f2e(g, v, i, j,   k) -
                         avgy_f2e(g, w, i, j, k+1) * avgz_f2e(g, v, i, j, k+1)
    end
end

@inline function u∇v(g::RegularCartesianGrid, U::VelocityFields, i, j, k)
    (δx_e2f_ūʸv̄ˣ(g, U.u, U.v, i, j, k) / g.Δx) + (δy_c2f_v̄ʸv̄ʸ(g, U.v, i, j, k) / g.Δy) + (δz_e2f_w̄ʸv̄ᶻ(g, U.v, U.w, i, j, k) / g.Δz)
end
