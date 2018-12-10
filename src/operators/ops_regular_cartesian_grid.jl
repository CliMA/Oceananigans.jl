using Oceananigans:
    RegularCartesianGrid,
    CellField, FaceField, FaceFieldX, FaceFieldY, FaceFieldZ,
    VelocityFields, TracerFields, PressureFields, SourceTerms, ForcingFields, TemporaryFields

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

"""
    δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)

Compute the difference \$\\delta_x(f) = f_E - f_W\$ between the eastern and
western cells of a cell-centered field `f` and store it in a face-centered
field `δxf`, assuming both fields are defined on a regular Cartesian grid `g`
with periodic boundary condition in the \$x\$-direction.
"""
function δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δxf.data[i, j, k] =  f.data[i, j, k] - f.data[decmod1(i, g.Nx), j, k]
    end
    nothing
end

"""
    δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)

Compute the difference \$\\delta_x(f) = f_E - f_W\$ between the eastern and
western faces of a face-centered field `f` and store it in a cell-centered
field `δxf`, assuming both fields are defined on a regular Cartesian grid `g`
with periodic boundary conditions in the \$x\$-direction.
"""
function δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δxf.data[i, j, k] =  f.data[incmod1(i, g.Nx), j, k] - f.data[i, j, k]
    end
    nothing
end

"""
    δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField)

Compute the difference \$\\delta_y(f) = f_N - f_S\$ between the northern and
southern cells of a cell-centered field `f` and store it in a face-centered
field `δyf`, assuming both fields are defined on a regular Cartesian grid `g`
with periodic boundary condition in the \$y\$-direction.
"""
function δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δyf.data[i, j, k] =  f.data[i, j, k] - f.data[i, decmod1(j, g.Ny), k]
    end
    nothing
end

"""
    δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField)

Compute the difference \$\\delta_y(f) = f_N - f_S\$ between the northern and
southern faces of a face-centered field `f` and store it in a cell-centered
field `δyf`, assuming both fields are defined on a regular Cartesian grid `g`
with periodic boundary condition in the \$y\$-direction.
"""
function δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δyf.data[i, j, k] =  f.data[i, incmod1(j, g.Ny), k] - f.data[i, j, k]
    end
    nothing
end

"""
    δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField)

Compute the difference \$\\delta_z(f) = f_T - f_B\$ between the top and
bottom cells of a cell-centered field `f` and store it in a face-centered
field `δzf`, assuming both fields are defined on a regular Cartesian grid `g`
with Neumann boundary condition in the \$z\$-direction.
"""
function δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField)
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δzf.data[i, j, k] = f.data[i, j, k-1] - f.data[i, j, k]
    end
    @. δzf.data[:, :, 1] = 0
    nothing
end

"""
    δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField)

Compute the difference \$\\delta_z(f) = f_T - f_B\$ between the top and
bottom faces of a face-centered field `f` and store it in a cell-centered
field `δzf`, assuming both fields are defined on a regular Cartesian grid `g`
with Neumann boundary condition in the \$z\$-direction.
"""
function δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField)
    for k in 1:(g.Nz-1), j in 1:g.Ny, i in 1:g.Nx
        @inbounds δzf.data[i, j, k] =  f.data[i, j, k] - f.data[i, j, k+1]
    end
    @. δzf.data[:, :, end] = f.data[:, :, end]
    nothing
end

"""
    avgx(g::RegularCartesianGrid, f::CellField, favgx::FaceField)

Compute the average \$\\overline{\\;f\\;}^x = \\frac{f_E + f_W}{2}\$ between the
eastern and western cells of a cell-centered field `f` and store it in a `g`
face-centered field `favgx`, assuming both fields are defined on a regular
Cartesian grid `g` with periodic boundary conditions in the \$x\$-direction.
"""
function avgx!(g::RegularCartesianGrid, f::CellField, favgx::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgx.data[i, j, k] =  (f.data[i, j, k] + f.data[decmod1(i, g.Nx), j, k]) / 2
    end
end

function avgx!(g::RegularCartesianGrid, f::FaceField, favgx::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgx.data[i, j, k] =  (f.data[incmod1(i, g.Nx), j, k] + f.data[i, j, k]) / 2
    end
end

function avgy!(g::RegularCartesianGrid, f::CellField, favgy::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgy.data[i, j, k] =  (f.data[i, j, k] + f.data[i, decmod1(j, g.Ny), k]) / 2
    end
end

function avgy!(g::RegularCartesianGrid, f::FaceField, favgy::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgy.data[i, j, k] =  (f.data[i, incmod1(j, g.Ny), k] + f.data[i, j, k]) / 2
    end
end

function avgz!(g::RegularCartesianGrid, f::CellField, favgz::FaceField)
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgz.data[i, j, k] =  (f.data[i, j, k] + f.data[i, j, k-1]) / 2
    end
    @. favgz.data[:, :, 1] = 0
    nothing
end

function avgz!(g::RegularCartesianGrid, f::FaceField, favgz::CellField)
    for k in 1:(g.Nz-1), j in 1:g.Ny, i in 1:g.Nx
        favgz.data[i, j, k] =  (f.data[i, j, incmod1(k, g.Nz)] + f.data[i, j, k]) / 2
    end
    @. favgz.data[:, :, end] = 0
    nothing
end

"""
    div!(g, fx, fy, fz, δfx, δfy, δfz, div)

Compute the divergence.
"""
function div!(g::RegularCartesianGrid,
              fx::FaceFieldX, fy::FaceFieldY, fz::FaceFieldZ, div::CellField,
              tmp::TemporaryFields)

    δxfx, δyfy, δzfz = tmp.fC1, tmp.fC2, tmp.fC3

    δx!(g, fx, δxfx)
    δy!(g, fy, δyfy)
    δz!(g, fz, δzfz)

    @. div.data = (1/g.V) * (g.Ax * δxfx.data + g.Ay * δyfy.data + g.Az * δzfz.data)
    nothing
end

function div!(g::RegularCartesianGrid,
              fx::CellField, fy::CellField, fz::CellField, div::FaceField,
              tmp::TemporaryFields)

    δxfx, δyfy, δzfz = tmp.fFX, tmp.fFY, tmp.fFZ

    δx!(g, fx, δxfx)
    δy!(g, fy, δyfy)
    δz!(g, fz, δzfz)

    @. div.data = (1/g.V) * (g.Ax * δxfx.data + g.Ay * δyfy.data + g.Az * δzfz.data)
    nothing
end

function div_flux!(g::RegularCartesianGrid,
                   u::FaceFieldX, v::FaceFieldY, w::FaceFieldZ, Q::CellField,
                   div_flux::CellField, tmp::TemporaryFields)

    Q̅ˣ, Q̅ʸ, Q̅ᶻ = tmp.fFX, tmp.fFY, tmp.fFZ

    avgx!(g, Q, Q̅ˣ)
    avgy!(g, Q, Q̅ʸ)
    avgz!(g, Q, Q̅ᶻ)

    flux_x, flux_y, flux_z = tmp.fFX, tmp.fFY, tmp.fFZ

    @. flux_x.data = g.Ax * u.data * Q̅ˣ.data
    @. flux_y.data = g.Ay * v.data * Q̅ʸ.data
    @. flux_z.data = g.Az * w.data * Q̅ᶻ.data

    # Imposing zero vertical flux through the top layer.
    @. flux_z.data[:, :, 1] = 0

    δxflux_x, δyflux_y, δzflux_z = tmp.fC1, tmp.fC2, tmp.fC3

    δx!(g, flux_x, δxflux_x)
    δy!(g, flux_y, δyflux_y)
    δz!(g, flux_z, δzflux_z)

    @. div_flux.data = (1/g.V) * (δxflux_x.data + δyflux_y.data + δzflux_z.data)
    nothing
end

function u∇u!(g::RegularCartesianGrid, ũ::VelocityFields, u∇u::FaceFieldX,
              tmp::TemporaryFields)

    u̅ˣ = tmp.fC1
    avgx!(g, ũ.u, u̅ˣ)

    uu = tmp.fC1
    @. uu.data = g.Ax * u̅ˣ.data^2

    u̅ʸ, v̅ˣ = tmp.fC2, tmp.fC3
    avgy!(g, ũ.u, u̅ʸ)
    avgx!(g, ũ.v, v̅ˣ)

    uv = tmp.fC2
    @. uv.data = g.Ay * u̅ʸ.data * v̅ˣ.data

    u̅ᶻ, w̅ˣ = tmp.fC3, tmp.fC4
    avgz!(g, ũ.u, u̅ᶻ)
    avgx!(g, ũ.w, w̅ˣ)

    uw = tmp.fC3
    @. uw.data = g.Az * u̅ᶻ.data * w̅ˣ.data

    ∂uu∂x, ∂uv∂y, ∂uw∂z = tmp.fFX, tmp.fFY, tmp.fFZ
    δx!(g, uu, ∂uu∂x)
    δy!(g, uv, ∂uv∂y)
    δz!(g, uw, ∂uw∂z)

    @. u∇u.data = (1/g.V) * (∂uu∂x.data + ∂uv∂y.data + ∂uw∂z.data)
    nothing
end

function u∇v!(g::RegularCartesianGrid, ũ::VelocityFields, u∇v::FaceFieldY,
              tmp::TemporaryFields)

    v̅ʸ = tmp.fC1
    avgy!(g, ũ.v, v̅ʸ)

    vv = tmp.fC1
    @. vv.data = g.Ay * v̅ʸ.data^2

    v̅ˣ, u̅ʸ = tmp.fC2, tmp.fC3
    avgx!(g, ũ.v, v̅ˣ)
    avgy!(g, ũ.u, u̅ʸ)

    vu = tmp.fC2
    @. vu.data = g.Ax * v̅ˣ.data * u̅ʸ.data

    v̅ᶻ, w̅ʸ = tmp.fC3, tmp.fC4
    avgz!(g, ũ.v, v̅ᶻ)
    avgy!(g, ũ.w, w̅ʸ)

    vw = tmp.fC3
    @. vw.data = g.Az * v̅ᶻ.data * w̅ʸ.data

    ∂vu∂x, ∂vv∂y, ∂vw∂z = tmp.fFX, tmp.fFY, tmp.fFZ
    δx!(g, vu, ∂vu∂x)
    δy!(g, vv, ∂vv∂y)
    δz!(g, vw, ∂vw∂z)

    @. u∇v.data = (1/g.V) * (∂vu∂x.data + ∂vv∂y.data + ∂vw∂z.data)
    nothing
end

function u∇w!(g::RegularCartesianGrid, ũ::VelocityFields, u∇w::FaceFieldZ,
              tmp::TemporaryFields)

    w̅ᶻ = tmp.fC1
    avgz!(g, ũ.w, w̅ᶻ)

    ww = tmp.fC1
    @. ww.data = g.Ay * w̅ᶻ.data^2

    @. ww.data[:, :, 1]   .= 0
    @. ww.data[:, :, end] .= 0

    w̅ˣ, u̅ᶻ = tmp.fC2, tmp.fC3
    avgx!(g, ũ.w, w̅ˣ)
    avgz!(g, ũ.u, u̅ᶻ)

    wu = tmp.fC2
    @. wu.data = g.Ax * w̅ˣ.data * u̅ᶻ.data

    w̅ʸ, v̅ᶻ = tmp.fC3, tmp.fC4
    avgy!(g, ũ.w, w̅ʸ)
    avgz!(g, ũ.v, v̅ᶻ)

    wv = tmp.fC3
    @. wv.data = g.Az * w̅ʸ.data * v̅ᶻ.data

    ∂wu∂x, ∂wv∂y, ∂ww∂z = tmp.fFX, tmp.fFY, tmp.fFZ
    δx!(g, wu, ∂wu∂x)
    δy!(g, wv, ∂wv∂y)
    δz!(g, ww, ∂ww∂z)

    @. u∇w.data = (1/g.V) * (∂wu∂x.data + ∂wv∂y.data + ∂ww∂z.data)
    nothing
end

function κ∇²!(g::RegularCartesianGrid, Q::CellField, κ∇²Q::CellField, κh, κv,
             tmp::TemporaryFields)
    δxQ, δyQ, δzQ = tmp.fFX, tmp.fFY, tmp.fFZ

    δx!(g, Q, δxQ)
    δy!(g, Q, δyQ)
    δz!(g, Q, δzQ)

    κ∇Q_x, κ∇Q_y, κ∇Q_z = tmp.fFX, tmp.fFY, tmp.fFZ

    @. κ∇Q_x.data = κh * δxQ.data / g.Δx
    @. κ∇Q_y.data = κh * δyQ.data / g.Δy
    @. κ∇Q_z.data = κv * δzQ.data / g.Δz

    div!(g, κ∇Q_x, κ∇Q_y, κ∇Q_z, κ∇²Q, tmp)
    nothing
end
