using Oceananigans: RegularCartesianGrid, CellField, FaceField

# Increment and decrement integer a with periodic wrapping. So if n == 10 then
# incmod1(11, n) = 1 and decmod1(0, n) = 10.
@inline incmod1(a, n) = a == n ? one(a) : a + 1
@inline decmod1(a, n) = a == 1 ? n : a - 1

function δx!(g::RegularCartesianGrid, f::CellField, δxf::FaceField)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds δxf[i, j, k] =  f[i, j, k] - f[decmod1(i,Nx), j, k]
    end
end

function δx!(g::RegularCartesianGrid, f::FaceField, δxf::CellField)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        @inbounds δxf[i, j, k] =  f[incmod1(i, Nx), j, k] - f[i, j, k]
    end
end
