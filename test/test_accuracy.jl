using Oceananigans
using Oceananigans.Models: ShallowWaterModel
using Oceananigans.Grids: Periodic, Bounded
using Oceananigans.Advection
using Oceananigans.Grids: xnodes,ynodes

using Plots

include("../src/Models/ShallowWaterModels/shallow_water_advection_operators.jl")
using Oceananigans.Operators



amp   = 1e-1
L     = 10
width = 0.3

Nrange = 2 .^collect(6:10)
error  = zeros(length(Nrange));


for i in collect(1:length(Nrange))

    N = Nrange[i]
    
    grid = RegularCartesianGrid(size=(N, 1, 1), x=(-L/2,L/2), y=(0,1), z=(0,1), topology=(Periodic, Periodic, Bounded));
    
    model = ShallowWaterModel(        grid = grid,
            gravitational_acceleration = 1,
                          architecture = CPU(),
                             advection = WENO5(), 
                              coriolis = FPlane(f=0.0)
                                  )
    
    h(x, y, z) = 1.0 + amp * exp(- x^2 / (2width^2));  
    set!(model, h = h)
    
    xc = xnodes(model.solution.h);
    xf = xc .- grid.Δx/2

    hf  = 1.0 .+ amp.*exp.( - xf.^2/(2width^2) );
    hc  = 1.0 .+ amp.*exp.( - xc.^2/(2width^2) );

    dhf = -amp/width^2*xf.*exp.( - xf.^2/(2width^2) );
    dhc = -amp/width^2*xc.*exp.( - xc.^2/(2width^2) );
    
    set!(model, h = h)
    
    dhdx = zeros(model.grid.Nx, model.grid.Ny, model.grid.Nz)
    for k in 1:model.grid.Nz, j in 1:model.grid.Ny, i in 1:model.grid.Nx
        dhdx[i,j,k] = ∂x₂ᶠᵃᵃ(i, j, k, grid, model.solution.h)    
    end
    
    diffdh = dhdx[:,1,1] - dhf
    #error[cnt] = diffdh[Int(N/2)];
    error[i] = sqrt(sum(diffdh[2:end].^2))
    println("N = ", N, " error is ", error[i])
    
end


plt1 = plot(Nrange, abs.(error), xaxis=:log, yaxis=:log, seriestype = :scatter)
plt2 = plot!(Nrange, 30*Nrange.^(-2), linewidth=3)
#display(plt2)
savefig("results")

for i in collect(1:length(Nrange)-1)
    println("i = ", i, " and error = ", error[i]/error[i+1])
end

