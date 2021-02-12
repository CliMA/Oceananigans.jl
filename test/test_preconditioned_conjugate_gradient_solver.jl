using Statistics

arch = archs[1]

@kernel function ∇²!(grid, f, ∇²f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²(i, j, k, grid, f)
end

@kernel function divergence!(grid, u, v, w, div)
    i, j, k = @index(Global, NTuple)
    @inbounds div[i, j, k] = divᶜᶜᶜ(i, j, k, grid, u, v, w)
end

@testset "Conjugate Gradient solvers" begin
    @info "Testing Conjugate Gradient solvers..."

function runtest()
Lx, Ly, Lz = 4e6, 6e6, 1
Nx, Ny, Nz = 100, 150, 1
grid = RegularCartesianGrid(size=(Nx,Ny,Nz), extent=(Lx,Ly,Lz) )


function Amatrix_function!(result,x,arch,grid,bcs)
 event = launch!(arch, grid, :xyz, ∇²!, grid, x, result, dependencies=Event(device(arch)))
 wait(device(arch), event)
 fill_halo_regions!(result,bcs,arch,grid)
end

# Fields for flow, divergence of flow, RHS, and potential to make non-divergent, ϕ
velocities = VelocityFields(arch, grid)
RHS        = CenterField(arch, grid)
ϕ          = CenterField(arch, grid)

# Set divergent flow and calculate divergence
(u, v, w)  = velocities
imid=Int(floor(grid.Nx/2))+1
jmid=Int(floor(grid.Ny/2))+1
u.data[imid,jmid,1]=1.
fill_halo_regions!(u.data,u.boundary_conditions,arch,grid)
event = launch!(arch, grid, :xyz, divergence!, grid, u.data, v.data, w.data, RHS.data,
                    dependencies=Event(device(arch)))
wait(device(arch), event)
fill_halo_regions!(RHS.data,RHS.boundary_conditions,arch,grid)

pcg_solver=PCGSolver( ;arch=arch,
              parameters=(PCmatrix_function=nothing,
                          Amatrix_function= Amatrix_function!,
                          Template_field=RHS,
                          maxit=grid.Nx*grid.Ny,
                          tol=1.e-13,
                         )
           )

# Set initial guess and solve
ϕ.data.=0.
solve_poisson_equation!(pcg_solver,RHS.data,ϕ.data)

# Compute ∇² of solution
result=similar(ϕ.data)
event = launch!(arch, grid, :xyz, ∇²!, grid, ϕ.data, result, dependencies=Event(device(arch)))
wait(device(arch), event)
fill_halo_regions!(result,ϕ.boundary_conditions,arch,grid)

mincheck=abs(minimum(result[:,:,1].-RHS.data[:,:,1])) < 1.e-12
maxcheck=abs(maximum(result[:,:,1].-RHS.data[:,:,1])) < 1.e-12
stdcheck=std(result[:,:,1].-RHS.data[:,:,1]) < 1.e-14

return mincheck & maxcheck & stdcheck

end

@test runtest()

end

## using Plots
## p1=heatmap(interior(u)[:,:,1],title='u')
## p2=heatmap(interior(RHS)[:,:,1],title="∇⋅U")
## p3=heatmap(interior(ϕ)[:,:,1],title="η")
## contour!(interior(ϕ)[:,:,1],linecolor=(:black))
## p4=heatmap(result[1:Nx,1:Ny,1].-RHS.data[1:Nx,1:Ny,1],title="∇²η - ∇⋅U")
## plot(p1,p2,p3,p4,size=(1600,1600))
## savefig("plot.png")

