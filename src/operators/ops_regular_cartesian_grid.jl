using Oceananigans: RegularCartesianGrid, CellField, FaceField

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
        @inbounds δxf[i, j, k] =  f[i, j, k] - f[decmod1(i, g.Nx), j, k]
    end
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
        @inbounds δxf[i, j, k] =  f[incmod1(i, g.Nx), j, k] - f[i, j, k]
    end
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
        @inbounds δyf[i, j, k] =  f[i, j, k] - f[i, decmod1(j, g.Ny), k]
    end
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
        @inbounds δyf[i, j, k] =  f[i, incmod1(j, g.Ny), k] - f[i, j, k]
    end
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
        @inbounds δzf[i, j, k] = f[i, j, k-1] - f[i, j, k]
    end
    @. δzf[:, :, 1] = 0
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
        @inbounds δf[i, j, k] =  f[i, j, k] - f[i, j, k+1]
    end
    @. δf[:, :, end] = f[:, :, end]
end

"""
    avgx(g::RegularCartesianGrid, f::CellField, favgx::FaceField)

Compute the average \$\\overline{\\;f\\;}^x = \\frac{f_E + f_W}{2}\$ between the
eastern and western cells of a cell-centered field `f` and store it in a `g`
face-centered field `favgx`, assuming both fields are defined on a regular
Cartesian grid `g` with periodic boundary conditions in the \$x\$-direction.
"""
function avgx(g::RegularCartesianGrid, f::CellField, favgx::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgx[i, j, k] =  (f[i, j, k] + f[decmod1(i, g.Nx), j, k]) / 2
    end
end

function avgx(g::RegularCartesianGrid, f::FaceField, favgx::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgx[i, j, k] =  (f[incmod1(i, g.Nx), j, k] + f[i, j, k]) / 2
    end
end

function avgy(g::RegularCartesianGrid, f::CellField, favgx::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgy[i, j, k] =  (f[i, j, k] + f[i, decmod1(j, g.Ny), k]) / 2
    end
end

function avgy(g::RegularCartesianGrid, f::FaceField, favgx::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds favgy[i, j, k] =  (f[i, incmod1(j, g.Ny), k] + f[i, j, k]) / 2
    end
end

function avgz(g::RegularCartesianGrid, f::CellField, favgz::FaceField)
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        favgz[i, j, k] =  (f[i, j, k] + f[i, j, k-1]) / 2
    end
    @. favgz[:, :, 1] = 0
end

function avgz(g::RegularCartesianGrid, f::FaceField, favgz::CellField)
    for k in 1:(g.Nz-1), j in 1:g.Ny, i in 1:g.Nx
        favgz[i, j, k] =  (f[i, j, incmod1(k, g.Nz)] + f[i, j, k]) / 2
    end
    @. δf[:, :, end] = 0
end
