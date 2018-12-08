using Oceananigans: RegularCartesianGrid, CellField, FaceField

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

function δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δxf[i, j, k] =  f[i, j, k] - f[decmod1(i, g.Nx), j, k]
    end
end

function δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δxf[i, j, k] =  f[incmod1(i, g.Nx), j, k] - f[i, j, k]
    end
end

function δy!(g::RegularCartesianGrid, f::CellField, δyf::FaceField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δyf[i, j, k] =  f[i, j, k] - f[i, decmod1(j, g.Ny), k]
    end
end

function δy!(g::RegularCartesianGrid, f::FaceField, δyf::CellField)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δyf[i, j, k] =  f[i, incmod1(j, g.Ny), k] - f[i, j, k]
    end
end

function δz!(g::RegularCartesianGrid, f::CellField, δzf::FaceField)
    for k in 2:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        @inbounds δzf[i, j, k] = f[i, j, k-1] - f[i, j, k]
    end
    @. δzf[:, :, 1] = 0
end

function δz!(g::RegularCartesianGrid, f::FaceField, δzf::CellField)
    for k in 1:(g.Nz-1), j in 1:g.Ny, i in 1:g.Nx
        @inbounds δf[i, j, k] =  f[i, j, k] - f[i, j, k+1]
    end
    @. δf[:, :, end] = f[:, :, end]
end
