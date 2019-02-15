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

@inline δx_c2f(f, Nx, i, j, k) = @inbounds f[i, j, k] - f[decmod1(i, Nx), j, k]
@inline δx_f2c(f, Nx, i, j, k) = @inbounds f[incmod1(i, Nx), j, k] - f[i, j, k]
@inline δx_e2f(f, Nx, i, j, k) = @inbounds f[incmod1(i, Nx), j, k] - f[i, j, k]
@inline δx_f2e(f, Nx, i, j, k) = @inbounds f[i, j, k] - f[decmod1(i, Nx), j, k]

# @inline δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])
# @inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
# @inline δx!(g::RegularCartesianGrid, f::EdgeField, δxf::FaceField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k])
# @inline δx!(g::RegularCartesianGrid, f::FaceField, δxf::EdgeField, i, j, k) = (@inbounds δxf.data[i, j, k] = f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k])

@inline δy_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k]
@inline δy_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k]
@inline δy_e2f(g::RegularCartesianGrid, f::EdgeField, i, j, k) = @inbounds f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k]
@inline δy_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k]

@inline δy_c2f(f, Ny, i, j, k) = @inbounds f[i, j, k] - f[i, decmod1(j, Ny), k]
@inline δy_f2c(f, Ny, i, j, k) = @inbounds f[i, incmod1(j, Ny), k] - f[i, j, k]
@inline δy_e2f(f, Ny, i, j, k) = @inbounds f[i, incmod1(j, Ny), k] - f[i, j, k]
@inline δy_f2e(f, Ny, i, j, k) = @inbounds f[i, j, k] - f[i, decmod1(j, Ny), k]

# @inline δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])
# @inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
# @inline δy!(g::RegularCartesianGrid, f::EdgeField, δyf::FaceField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k])
# @inline δy!(g::RegularCartesianGrid, f::FaceField, δyf::EdgeField, i, j, k) = (@inbounds δyf.data[i, j, k] = f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k])

@inline function δz_c2f(g::RegularCartesianGrid, f::CellField, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f.data[i, j, k-1] - f.data[i, j, k]
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

@inline function δz_c2f(f, Nz, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

@inline function δz_f2c(f, Nz, i, j, k)
    if k == Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_e2f(f, Nz, i, j, k)
    if k == Nz
        @inbounds return f[i, j, k]
    else
        @inbounds return f[i, j, k] - f[i, j, k+1]
    end
end

@inline function δz_f2e(f, Nz, i, j, k)
    if k == 1
        return 0
    else
        @inbounds return f[i, j, k-1] - f[i, j, k]
    end
end

# @inline function δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField, i, j, k)
#     if k == 1
#         @inbounds δzf.data[i, j, k] = 0
#     else
#         @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
#     end
# end

# @inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField, i, j, k)
#     if k == g.Nz
#         @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
#     else
#         @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
#     end
# end
#
# @inline function δz!(g::RegularCartesianGrid, f::EdgeField, δzf::FaceField, i, j, k)
#     if k == g.Nz
#         @inbounds δzf.data[i, j, g.Nz] = f.data[i, j, g.Nz]
#     else
#         @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
#     end
# end
#
# @inline function δz!(g::RegularCartesianGrid, f::FaceField, δzf::EdgeField, i, j, k)
#     if k == 1
#         @inbounds δzf.data[i, j, k] = 0
#     else
#         @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
#     end
# end

@inline avgx_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k])
@inline avgx_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[incmod1(i, g.Nx), j, k] + f.data[i, j, k])
@inline avgx_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k])

@inline avgx_c2f(f, Nx, i, j, k) = @inbounds 0.5f0 * (f[i, j, k] + f[decmod1(i, Nx), j, k])
@inline avgx_f2c(f, Nx, i, j, k) = @inbounds 0.5f0 * (f[incmod1(i, Nx), j, k] + f[i, j, k])
@inline avgx_f2e(f, Nx, i, j, k) = @inbounds 0.5f0 * (f[i, j, k] + f[decmod1(i, Nx), j, k])

# @inline avgx!(g::RegularCartesianGrid, f::CellField, favgx::FaceField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]))
# @inline avgx!(g::RegularCartesianGrid, f::FaceField, favgx::CellField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[incmod1(i, g.Nx), j, k] + f.data[i, j, k]))
# @inline avgx!(g::RegularCartesianGrid, f::FaceField, favgx::EdgeField, i, j, k) = (@inbounds favgx.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]))

@inline avgy_c2f(g::RegularCartesianGrid, f::CellField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k])
@inline avgy_f2c(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, incmod1(j, g.Ny), k] + f.data[i, j, k])
@inline avgy_f2e(g::RegularCartesianGrid, f::FaceField, i, j, k) = @inbounds 0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k])

@inline avgy_c2f(f, Ny, i, j, k) = @inbounds 0.5f0 * (f[i, j, k] + f[i, decmod1(j, Ny), k])
@inline avgy_f2c(f, Ny, i, j, k) = @inbounds 0.5f0 * (f[i, incmod1(j, Ny), k] + f[i, j, k])
@inline avgy_f2e(f, Ny, i, j, k) = @inbounds 0.5f0 * (f[i, j, k] + f[i, decmod1(j, Ny), k])

# @inline avgy!(g::RegularCartesianGrid, f::CellField, favgy::FaceField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]))
# @inline avgy!(g::RegularCartesianGrid, f::FaceField, favgy::CellField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, incmod1(j, g.Ny), k] + f.data[i, j, k]))
# @inline avgy!(g::RegularCartesianGrid, f::FaceField, favgy::EdgeField, i, j, k) = (@inbounds favgy.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]))

@inline avg_xy(g::RegularCartesianGrid, u::FaceFieldX, i, j, k) = @inbounds 0.5f0 * (avgy_f2c(g, u, i, j, k) + avg_f2c(g, u, incmod1(i, g.Nx), j, k))

@inline avg_xy(u, Nx, Ny, i, j, k) = @inbounds 0.5f0 * (avgy_f2c(u, Ny, i, j, k) + avg_f2c(u, Nx, incmod1(i, Nx), j, k))

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

@inline function avgz_c2f(f, Nz, i, j, k)
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return  0.5f0 * (f[i, j, k] + f[i, j, k-1])
    end
end

@inline function avgz_f2c(f, Nz, i, j, k)
    if k == Nz
        @inbounds return 0.5f0 * f[i, j, k]
    else
        @inbounds return 0.5f0 * (f[i, j, k+1] + f[i, j, k])
    end
end

@inline function avgz_f2e(f, Nz, i, j, k)
    if k == 1
        @inbounds return f[i, j, k]
    else
        @inbounds return 0.5f0 * (f[i, j, k] + f[i, j, k-1])
    end
end

# @inline function avgz!(g::RegularCartesianGrid, f::CellField, favgz::FaceField, i, j, k)
#     if k == 1
#         @inbounds favgz.data[i, j, k] = f.data[i, j, k]
#     else
#         @inbounds favgz.data[i, j, k] =  0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
#     end
# end
#
# @inline function avgz!(g::RegularCartesianGrid, f::FaceField, favgz::CellField, i, j, k)
#     if k == g.Nz
#         @inbounds favgz.data[i, j, k] = 0.5f0 * f.data[i, j, k]
#     else
#         @inbounds favgz.data[i, j, k] = 0.5f0 * (f.data[i, j, incmod1(k, g.Nz)] + f.data[i, j, k])
#     end
# end
#
# @inline function avgz!(g::RegularCartesianGrid, f::FaceField, favgz::EdgeField, i, j, k)
#     if k == 1
#         @inbounds favgz.data[i, j, k] = f.data[i, j, k]
#     else
#         @inbounds favgz.data[i, j, k] = 0.5f0 * (f.data[i, j, k] + f.data[i, j, k-1])
#     end
# end

@inline function div(g::RegularCartesianGrid, fx::FaceFieldX, fy::FaceFieldY, fz::FaceFieldZ, i, j, k)
    (δx_f2c(g, fx, i, j, k) / g.Δx) + (δy_f2c(g, fy, i, j, k) / g.Δy) + (δz_f2c(g, fz, i, j, k) / g.Δz)
end

@inline function div_f2c(fx, fy, fz, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    (δx_f2c(fx, Nx, i, j, k) / Δx) + (δy_f2c(fy, Ny, i, j, k) / Δy) + (δz_f2c(fz, Nz, i, j, k) / Δz)
end

@inline function div(g::RegularCartesianGrid, fx::CellField, fy::CellField, fz::CellField, i, j, k)
    (δx_c2f(g, fx, i, j, k) / g.Δx) + (δy_c2f(g, fy, i, j, k) / g.Δy) + (δz_c2f(g, fz, i, j, k) / g.Δz)
end

@inline function div_c2f(fx, fy, fz, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    (δx_c2f(fx, Nx, i, j, k) / Δx) + (δy_c2f(fy, Ny, i, j, k) / Δy) + (δz_c2f(fz, Nz, i, j, k) / Δz)
end

@inline function δx_f2c_ab̄ˣ(g::RegularCartesianGrid, a::FaceFieldX, b::CellField, i, j, k)
    @inbounds (a.data[incmod1(i, g.Nx), j, k] * avgx_c2f(g, b, incmod1(i, g.Nx), j, k) -
               a.data[i,                j, k] * avgx_c2f(g, b, i,                j, k))
end

@inline function δx_f2c_ab̄ˣ(a, b, Nx, i, j, k)
    @inbounds (a[incmod1(i, Nx), j, k] * avgx_c2f(b, Nx, incmod1(i, Nx), j, k) -
               a[i,              j, k] * avgx_c2f(b, Nx, i,              j, k))
end

@inline function δy_f2c_ab̄ʸ(g::RegularCartesianGrid, a::FaceFieldY, b::CellField, i, j, k)
    @inbounds (a.data[i, incmod1(j, g.Ny), k] * avgy_c2f(g, b, i, incmod1(j, g.Ny), k) -
               a.data[i,                j, k] * avgy_c2f(g, b, i, j,                k))
end

@inline function δy_f2c_ab̄ʸ(a, b, Ny, i, j, k)
    @inbounds (a[i, incmod1(j, Ny), k] * avgy_c2f(g, b, i, incmod1(j, Ny), k) -
               a[i,              j, k] * avgy_c2f(g, b, i, j,              k))
end

@inline function δz_f2c_ab̄ᶻ(g::RegularCartesianGrid, a::FaceFieldZ, b::CellField, i, j, k)
    if k == g.Nz
        @inbounds return a.data[i, j, k] * avgz_c2f(g, b, i, j, k)
    else
        @inbounds return (a.data[i, j,   k] * avgz_c2f(g, b, i, j,   k) -
                          a.data[i, j, k+1] * avgz_c2f(g, b, i, j, k+1))
    end
end

@inline function δz_f2c_ab̄ᶻ(a, b, Nz, i, j, k)
    if k == Nz
        @inbounds return a[i, j, k] * avgz_c2f(g, b, i, j, k)
    else
        @inbounds return (a[i, j,   k] * avgz_c2f(g, b, i, j,   k) -
                          a[i, j, k+1] * avgz_c2f(g, b, i, j, k+1))
    end
end

@inline function div_flux(g::RegularCartesianGrid, U::VelocityFields, Q::CellField, i, j, k)
    if k == 1
        @inbounds return (δx_f2c_ab̄ˣ(g, U.u, Q, i, j, k) / g.Δx) + (δy_f2c_ab̄ʸ(g, U.v, Q, i, j, k) / g.Δy) - ((U.w.data[i, j, 2] * avgz_c2f(g, Q, i, j, 2)) / g.Δz)
    else
        return (δx_f2c_ab̄ˣ(g, U.u, Q, i, j, k) / g.Δx) + (δy_f2c_ab̄ʸ(g, U.v, Q, i, j, k) / g.Δy) + (δz_f2c_ab̄ᶻ(g, U.w, Q, i, j, k) / g.Δz)
    end
end

@inline function div_flux(u, v, w, Q, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    if k == 1
        @inbounds return (δx_f2c_ab̄ˣ(u, Q, Nx, i, j, k) / Δx) + (δy_f2c_ab̄ʸ(v, Q, Ny, i, j, k) / Δy) - ((w[i, j, 2] * avgz_c2f(Q, Nz, i, j, 2)) / Δz)
    else
        return (δx_f2c_ab̄ˣ(u, Q, Nx, i, j, k) / Δx) + (δy_f2c_ab̄ʸ(v, Q, Ny, i, j, k) / Δy) + (δz_f2c_ab̄ᶻ(w, Q, Nz, i, j, k) / Δz)
    end
end

@inline function δx_c2f_ūˣūˣ(g::RegularCartesianGrid, u::FaceFieldX, i, j, k)
    avgx_f2c(g, u, i, j, k)^2 - avgx_f2c(g, u, decmod1(i, g.Nx), j, k)^2
end

@inline function δx_c2f_ūˣūˣ(u, Nx, i, j, k)
    avgx_f2c(u, Nx, i, j, k)^2 - avgx_f2c(u, Nx, decmod1(i, Nx), j, k)^2
end

@inline function δy_e2f_v̄ˣūʸ(g::RegularCartesianGrid, u::FaceFieldX, v::FaceFieldY, i, j, k)
    avgx_f2e(g, v, i, incmod1(j, g.Ny), k) * avgy_f2e(g, u, i, incmod1(j, g.Ny), k) -
    avgx_f2e(g, v, i,                j, k) * avgy_f2e(g, u, i,                j, k)
end

@inline function δy_e2f_v̄ˣūʸ(u, v, Nx, Ny, i, j, k)
    avgx_f2e(v, Nx, i, incmod1(j, Ny), k) * avgy_f2e(u, Ny, i, incmod1(j, Ny), k) -
    avgx_f2e(v, Nx, i,              j, k) * avgy_f2e(u, Ny, i,              j, k)
end

@inline function δz_e2f_w̄ˣūᶻ(g::RegularCartesianGrid, u::FaceFieldX, w::FaceFieldZ, i, j, k)
    if k == g.Nz
        @inbounds return avgx_f2e(g, w, i, j, k) * avgz_f2e(g, u, i, j, k)
    else
        @inbounds return avgx_f2e(g, w, i, j,   k) * avgz_f2e(g, u, i, j,   k) -
                         avgx_f2e(g, w, i, j, k+1) * avgz_f2e(g, u, i, j, k+1)
    end
end

@inline function δz_e2f_w̄ˣūᶻ(u, w, Nx, Nz, i, j, k)
    if k == Nz
        @inbounds return avgx_f2e(w, Nx, i, j, k) * avgz_f2e(u, Nx, i, j, k)
    else
        @inbounds return avgx_f2e(w, Nx, i, j,   k) * avgz_f2e(u, Nz, i, j,   k) -
                         avgx_f2e(w, Nx, i, j, k+1) * avgz_f2e(u, Nz, i, j, k+1)
    end
end

@inline function u∇u(g::RegularCartesianGrid, U::VelocityFields, i, j, k)
    (δx_c2f_ūˣūˣ(g, U.u, i, j, k) / g.Δx) + (δy_e2f_v̄ˣūʸ(g, U.u, U.v, i, j, k) / g.Δy) + (δz_e2f_w̄ˣūᶻ(g, U.u, U.w, i, j, k) / g.Δz)
end

@inline function u∇u(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    (δx_c2f_ūˣūˣ(u, Nx, i, j, k) / Δx) + (δy_e2f_v̄ˣūʸ(u, v, Nx, Ny, i, j, k) / Δy) + (δz_e2f_w̄ˣūᶻ(u, w, Nx, Nz, i, j, k) / Δz)
end

@inline function δx_e2f_ūʸv̄ˣ(g::RegularCartesianGrid, u::FaceFieldX, v::FaceFieldY, i, j, k)
    avgy_f2e(g, u, incmod1(i, g.Nx), j, k) * avgx_f2e(g, v, incmod1(i, g.Nx), j, k) -
    avgy_f2e(g, u, i,                j, k) * avgx_f2e(g, v, i,                j, k)
end

@inline function δx_e2f_ūʸv̄ˣ(u, v, Nx, Ny, i, j, k)
    avgy_f2e(u, Ny, incmod1(i, Nx), j, k) * avgx_f2e(v, Nx, incmod1(i, Nx), j, k) -
    avgy_f2e(u, Ny, i,              j, k) * avgx_f2e(v, Nx, i,              j, k)
end

@inline function δy_c2f_v̄ʸv̄ʸ(g::RegularCartesianGrid, v::FaceFieldY, i, j, k)
    avgy_f2c(g, v, i, j, k)^2 - avgy_f2c(g, v, i, decmod1(j, g.Ny), k)^2
end

@inline function δy_c2f_v̄ʸv̄ʸ(v, Ny, i, j, k)
    avgy_f2c(v, Ny, i, j, k)^2 - avgy_f2c(v, Ny, i, decmod1(j, Ny), k)^2
end

@inline function δz_e2f_w̄ʸv̄ᶻ(g::RegularCartesianGrid, v::FaceFieldY, w::FaceFieldZ, i, j, k)
    if k == g.Nz
        @inbounds return avgy_f2e(g, w, i, j, k) * avgz_f2e(g, v, i, j, k)
    else
        @inbounds return avgy_f2e(g, w, i, j,   k) * avgz_f2e(g, v, i, j,   k) -
                         avgy_f2e(g, w, i, j, k+1) * avgz_f2e(g, v, i, j, k+1)
    end
end

@inline function δz_e2f_w̄ʸv̄ᶻ(v, w, Ny, Nz, i, j, k)
    if k == Nz
        @inbounds return avgy_f2e(w, Ny, i, j, k) * avgz_f2e(v, Nz, i, j, k)
    else
        @inbounds return avgy_f2e(w, Ny, i, j,   k) * avgz_f2e(v, Nz, i, j,   k) -
                         avgy_f2e(w, Ny, i, j, k+1) * avgz_f2e(v, Nz, i, j, k+1)
    end
end

@inline function u∇v(g::RegularCartesianGrid, U::VelocityFields, i, j, k)
    (δx_e2f_ūʸv̄ˣ(g, U.u, U.v, i, j, k) / g.Δx) + (δy_c2f_v̄ʸv̄ʸ(g, U.v, i, j, k) / g.Δy) + (δz_e2f_w̄ʸv̄ᶻ(g, U.v, U.w, i, j, k) / g.Δz)
end

@inline function u∇v(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    (δx_e2f_ūʸv̄ˣ(u, v, Nx, Ny, i, j, k) / Δx) + (δy_c2f_v̄ʸv̄ʸ(v, Ny, i, j, k) / Δy) + (δz_e2f_w̄ʸv̄ᶻ(v, w, Ny, Nz, i, j, k) / Δz)
end

@inline function δx_e2f_ūᶻw̄ˣ(g::RegularCartesianGrid, u::FaceFieldX, w::FaceFieldZ, i, j, k)
    avgz_f2e(g, u, incmod1(i, g.Nx), j, k) * avgx_f2e(g, w, incmod1(i, g.Nx), j, k) -
    avgz_f2e(g, u, i,                j, k) * avgx_f2e(g, w, i,                j, k)
end

@inline function δx_e2f_ūᶻw̄ˣ(u, w, Nx, Nz, i, j, k)
    avgz_f2e(u, Nz, incmod1(i, Nx), j, k) * avgx_f2e(w, Nx, incmod1(i, Nx), j, k) -
    avgz_f2e(u, Nz, i,              j, k) * avgx_f2e(w, Nx, i,              j, k)
end

@inline function δy_e2f_v̄ᶻw̄ʸ(g::RegularCartesianGrid, v::FaceFieldY, w::FaceFieldZ, i, j, k)
    avgz_f2e(g, v, i, incmod1(j, g.Ny), k) * avgy_f2e(g, w, i, incmod1(j, g.Ny), k) -
    avgz_f2e(g, v, i,                j, k) * avgy_f2e(g, w, i,                j, k)
end

@inline function δy_e2f_v̄ᶻw̄ʸ(v, w, Ny, Nz, i, j, k)
    avgz_f2e(v, Nz, i, incmod1(j, Ny), k) * avgy_f2e(w, Ny, i, incmod1(j, Ny), k) -
    avgz_f2e(v, Nz, i,              j, k) * avgy_f2e(w, Ny, i,              j, k)
end

@inline function δz_c2f_w̄ᶻw̄ᶻ(g::RegularCartesianGrid, w::FaceFieldZ, i, j, k)
    if k == 1
        return 0
    else
        return avgz_f2c(g, w, i, j, k-1)^2 - avgz_f2c(g, w, i, j, k)^2
    end
end

@inline function δz_c2f_w̄ᶻw̄ᶻ(w, Nz, i, j, k)
    if k == 1
        return 0
    else
        return avgz_f2c(w, Nz, i, j, k-1)^2 - avgz_f2c(w, Nz, i, j, k)^2
    end
end


@inline function u∇w(g::RegularCartesianGrid, U::VelocityFields, i, j, k)
    (δx_e2f_ūᶻw̄ˣ(g, U.u, U.w, i, j, k) / g.Δx) + (δy_e2f_v̄ᶻw̄ʸ(g, U.v, U.w, i, j, k) / g.Δy) + (δz_c2f_w̄ᶻw̄ᶻ(g, U.w, i, j, k) / g.Δz)
end

@inline function u∇w(u, v, w, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    (δx_e2f_ūᶻw̄ˣ(u, w, Nx, Nz, i, j, k) / Δx) + (δy_e2f_v̄ᶻw̄ʸ(v, w, Ny, Nz, i, j, k) / Δy) + (δz_c2f_w̄ᶻw̄ᶻ(w, Nz, i, j, k) / Δz)
end

@inline δx²_c2f2c(g::RegularCartesianGrid, f::CellField, i, j, k) = δx_c2f(g, f, incmod1(i, g.Nx), j, k) - δx_c2f(g, f, i, j, k)
@inline δy²_c2f2c(g::RegularCartesianGrid, f::CellField, i, j, k) = δy_c2f(g, f, i, incmod1(j, g.Ny), k) - δy_c2f(g, f, i, j, k)

@inline δx²_c2f2c(f, Nx, i, j, k) = δx_c2f(f, Nx, incmod1(i, Nx), j, k) - δx_c2f(f, Nx, i, j, k)
@inline δy²_c2f2c(f, Ny, i, j, k) = δy_c2f(f, Ny, i, incmod1(j, Ny), k) - δy_c2f(f, Ny, i, j, k)

@inline function δz²_c2f2c(g::RegularCartesianGrid, f::CellField, i, j, k)
    if k == g.Nz
        return δz_c2f(g, f, i, j, k)
    else
        return δz_c2f(g, f, i, j, k) - δz_c2f(g, f, i, j, k+1)
    end
end

@inline function δz²_c2f2c(f, Nz, i, j, k)
    if k == Nz
        return δz_c2f(f, Nz, i, j, k)
    else
        return δz_c2f(f, Nz, i, j, k) - δz_c2f(f, Nz, i, j, k+1)
    end
end

@inline function κ∇²(g::RegularCartesianGrid, Q::CellField, κh, κv, i, j, k)
    ((κh/g.Δx^2) * δx²_c2f2c(g, Q, i, j, k)) + ((κh/g.Δy^2) * δy²_c2f2c(g, Q, i, j, k)) + ((κv/g.Δz^2) * δz²_c2f2c(g, Q, i, j, k))
end

@inline function κ∇²(Q, κh, κv, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    ((κh/Δx^2) * δx²_c2f2c(Q, Nx, i, j, k)) + ((κh/Δy^2) * δy²_c2f2c(Q, Ny, i, j, k)) + ((κv/Δz^2) * δz²_c2f2c(Q, Nz, i, j, k))
end

@inline δx²_f2c2f(g::RegularCartesianGrid, f::FaceField, i, j, k) = δx_f2c(g, f, i, j, k) - δx_f2c(g, f, decmod1(i, g.Nx), j, k)
@inline δy²_f2c2f(g::RegularCartesianGrid, f::FaceField, i, j, k) = δy_f2c(g, f, i, j, k) - δy_f2c(g, f, i, decmod1(j, g.Ny), k)

@inline δx²_f2e2f(g::RegularCartesianGrid, f::FaceField, i, j, k) = δx_f2e(g, f, incmod1(i, g.Nx), j, k) - δx_f2e(g, f, i, j, k)
@inline δy²_f2e2f(g::RegularCartesianGrid, f::FaceField, i, j, k) = δy_f2e(g, f, i, incmod1(j, g.Ny), k) - δy_f2e(g, f, i, j, k)

@inline δx²_f2c2f(f, Nx, i, j, k) = δx_f2c(f, Nx, i, j, k) - δx_f2c(f, Nx, decmod1(i, Nx), j, k)
@inline δy²_f2c2f(f, Ny, i, j, k) = δy_f2c(f, Ny, i, j, k) - δy_f2c(f, Ny, i, decmod1(j, Ny), k)

@inline δx²_f2e2f(f, Nx, i, j, k) = δx_f2e(f, Nx, incmod1(i, Nx), j, k) - δx_f2e(f, Nx, i, j, k)
@inline δy²_f2e2f(f, Ny, i, j, k) = δy_f2e(f, Ny, i, incmod1(j, Ny), k) - δy_f2e(f, Ny, i, j, k)

@inline function δz²_f2e2f(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == g.Nz
        return δz_f2e(g, f, i, j, k)
    else
        return δz_f2e(g, f, i, j, k) - δz_f2e(g, f, i, j, k+1)
    end
end

@inline function δz²_f2c2f(g::RegularCartesianGrid, f::FaceField, i, j, k)
    if k == 1
        return 0
    else
        return δz_f2c(g, f, i, j, k-1) - δz_f2c(g, f, i, j, k)
    end
end

@inline function δz²_f2e2f(f, Nz, i, j, k)
    if k == Nz
        return δz_f2e(f, Nz, i, j, k)
    else
        return δz_f2e(f, Nz, i, j, k) - δz_f2e(f, Nz, i, j, k+1)
    end
end

@inline function δz²_f2c2f(f, Nz, i, j, k)
    if k == 1
        return 0
    else
        return δz_f2c(f, Nz, i, j, k-1) - δz_f2c(f, Nz, i, j, k)
    end
end

@inline function 𝜈∇²u(g::RegularCartesianGrid, u::FaceFieldX, 𝜈h, 𝜈v, i, j, k)
    ((𝜈h/g.Δx^2) * δx²_f2c2f(g, u, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2e2f(g, u, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2e2f(g, u, i, j, k))
end

@inline function 𝜈∇²v(g::RegularCartesianGrid, v::FaceFieldY, 𝜈h, 𝜈v, i, j, k)
    ((𝜈h/g.Δx^2) * δx²_f2e2f(g, v, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2c2f(g, v, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2e2f(g, v, i, j, k))
end

@inline function 𝜈∇²w(g::RegularCartesianGrid, w::FaceFieldZ, 𝜈h, 𝜈v, i, j, k)
    ((𝜈h/g.Δx^2) * δx²_f2e2f(g, w, i, j, k)) + ((𝜈h/g.Δy^2) * δy²_f2e2f(g, w, i, j, k)) + ((𝜈v/g.Δz^2) * δz²_f2c2f(g, w, i, j, k))
end

@inline function 𝜈∇²u(u, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    ((𝜈h/Δx^2) * δx²_f2c2f(u, Nx, i, j, k)) + ((𝜈h/Δy^2) * δy²_f2e2f(u, Ny, i, j, k)) + ((𝜈v/Δz^2) * δz²_f2e2f(u, Nz, i, j, k))
end

@inline function 𝜈∇²v(v, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    ((𝜈h/Δx^2) * δx²_f2e2f(v, Nx, i, j, k)) + ((𝜈h/Δy^2) * δy²_f2c2f(v, Ny, i, j, k)) + ((𝜈v/Δz^2) * δz²_f2e2f(v, Nz, i, j, k))
end

@inline function 𝜈∇²w(w, 𝜈h, 𝜈v, Nx, Ny, Nz, Δx, Δy, Δz, i, j, k)
    ((𝜈h/Δx^2) * δx²_f2e2f(w, Nx, i, j, k)) + ((𝜈h/Δy^2) * δy²_f2e2f(w, Ny, i, j, k)) + ((𝜈v/Δz^2) * δz²_f2c2f(w, Nz, i, j, k))
end
