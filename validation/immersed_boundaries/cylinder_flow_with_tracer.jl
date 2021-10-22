pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..", "..")) 


ENV["GKSwstype"] = "nul"

using Plots
using ForwardDiff
using Measures
using LinearAlgebra

using Statistics
using Printf

using Oceananigans
using Oceananigans.Fields: interpolate
using Oceananigans.BoundaryConditions
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBoundary

#####
##### The Steady State Cylinder
#####

const R = 1.      # radius
const Re = 40.    # Reynolds number, Re < 40 --> Steady State
const nu = 2*R/Re  # viscosity


"""
    run_cylinder_steadyflow(output_time_interval = 1, stop_time = 100, arch = CPU(), Nh = 250, ν = 1/20,
                    momentum_advection = UpwindBiasedFifthOrder(), radius = R)

Run the steady state cylinder validation experiment until `stop_time` using `momentum_advection`
scheme or formulation, with horizontal resolution `Nh`, viscosity `ν`, and cylinder radius `R`, on `arch`itecture.
"""

function run_cylinder_steadystate(; output_time_interval = 1, stop_time = 100, arch = CPU(), Nh = 250, ν = nu, 
                                    advection = UpwindBiasedFifthOrder(), radius = R)


    inside_cylinder(x, y, z) = (x^2 + y^2) <= radius # immersed solid

    underlying_grid = RectilinearGrid(size=(Nh, Int(3*Nh/2),1), halo=(3, 3, 3),
                                           x = (-10, 10), y=(-10, 20), z = (0,1),
                                           topology = (Periodic, Bounded, Bounded))

    immersed_grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBoundary(inside_cylinder))
    
    # boundary conditions: inflow and outflow in y
    v_bc = OpenBoundaryCondition(1.0)
    v_bcs = FieldBoundaryConditions(north = v_bc, south = v_bc)

    immersed_model = NonhydrostaticModel(architecture = arch,
                                         advection = advection,
                                         timestepper = :RungeKutta3,
                                         grid = immersed_grid,
                                         tracers = :mass,
                                         closure = IsotropicDiffusivity(ν=ν, κ=ν),
                                         boundary_conditions = (v=v_bcs,),
                                         coriolis = nothing,
                                         buoyancy = nothing)

    # ** Initial conditions **
    #
    # u: no flow
    # v: uniform flow of U∞ = 1, except inside cylinder (no flow)

    # Parameters
    U∞ = 1.0 # uniform flow
    
    # Total initial conditions
    uᵢ(x, y, z) = 0
    vᵢ(x, y, z) = U∞
    mᵢ(x, y, z) = 1.0
    #mᵢ(x, y, z) = ifelse(inside_cylinder(x,y,z), 0., 1.0)
    set!(immersed_model, u=uᵢ, v=vᵢ, mass=mᵢ)

    wall_clock = [time_ns()]

    function progress(sim)
        @info(@sprintf("Iter: %d, time: %.1f, Δt: %.3f, wall time: %s, max|v|: %.2f",
                       sim.model.clock.iteration,
                       sim.model.clock.time,
                       sim.Δt.Δt,
                       prettytime(1e-9 * (time_ns() - wall_clock[1])),
                       maximum(abs, sim.model.velocities.v.data.parent)))

        wall_clock[1] = time_ns()

        return nothing
    end

    @show experiment_name = "cylinder_tracer_Nh_$(Nh)_$(typeof(immersed_model.advection).name.wrapper)"

    wizard = TimeStepWizard(cfl=0.07, Δt=0.07 * underlying_grid.Δx, max_change=1.1, max_Δt=10.0, min_Δt=0.0001)

    simulation = Simulation(immersed_model, Δt=wizard, stop_time=stop_time, iteration_interval=100, progress=progress)

    # Output: primitive fields + computations
    u, v, w, pHY′, pNHS, mass  = merge(immersed_model.velocities, immersed_model.pressures, immersed_model.tracers)
    outputs = merge(immersed_model.velocities, immersed_model.pressures, immersed_model.tracers)
    
    data_path = experiment_name
 
    simulation.output_writers[:fields] =
            JLD2OutputWriter(immersed_model, outputs,
                             schedule = TimeInterval(output_time_interval),
                             prefix = data_path,
                             field_slicer = nothing,
                             force = true)

    @info "Running a simulation of an steady state cylinder..."

    start_time = time_ns()

    run!(simulation)
    return experiment_name 
end

"""
    visualize_cylinder_steadystate(experiment_name)

Visualize the steady state cylinder data associated with `experiment_name`.
"""

# defining a function to mark boundary in plots
function circle_shape(h, k, r)
            θ = LinRange(0,2*π,500)
            h.+r*sin.(θ),k.+r*cos.(θ)
end

function visualize_cylinder_steadystate(experiment_name)

    @info "Making a fun movie about steady state cylinder flow..."
     
    filepath = experiment_name * ".jld2"

    u_timeseries = FieldTimeSeries(filepath,  "u")

    v_timeseries = FieldTimeSeries(filepath,  "v")
    
    p_timeseries = FieldTimeSeries(filepath,  "pNHS")

    m_timeseries = FieldTimeSeries(filepath, "mass")

    regular_grid = p_timeseries.grid

    xu, yu, zu = nodes(u_timeseries)
    xv, yv, zv = nodes(v_timeseries)
    xp, yp, zp = nodes(p_timeseries)
    xm, ym, zm = nodes(m_timeseries)

    anim = @animate for (i, t) in enumerate(p_timeseries.times)

        @info "Plotting frame $i of $(length(p_timeseries.times))..."

        Nx, Ny, Nz = size(regular_grid)

        u = u_timeseries[i]
        v = v_timeseries[i]
        p = p_timeseries[i]
        m = m_timeseries[i]

        ui = interior(u)[:, :, 1]
        vi = interior(v)[:, :, 1]
        pin = interior(p)[:, :, 1] # don't want to confuse with pi
        mi = interior(m)[:, :, 1]
        Nx, Ny = size(ui)

        kwargs = Dict(:aspectratio => 1,
                      :linewidth => 0,
                      :colorbar => :true,
                      :ticks => nothing,
                      :clims => (-1, 1),
                      :xlims => (-regular_grid.Lx/2, regular_grid.Lx/2),
                      :ylims => (-regular_grid.Ly/3, 2*regular_grid.Ly/3))

        u_plot = heatmap(xu, yu, clamp.(ui, -1, 1)'; color = :balance, size = (500,700), 
                            guidefontsize = 14, titlefont=14, top_margin = -10.0mm, bottom_margin = -10.0mm,
                            legendfont = 10, tickfont = 8, kwargs...)
        plot!(circle_shape(0,0,1),seriestype=[:shape,],linecolor=:black,
                            legend=false, fillalpha=0)
        
        v_plot = heatmap(xv, yv, clamp.(vi, -1, 1)'; color = :balance, size = (500,700), 
                            guidefontsize = 14, titlefont=14, top_margin = -10.0mm, bottom_margin = -10.0mm,
                            legendfont = 10, tickfont = 8, kwargs...)
        plot!(circle_shape(0,0,1),seriestype=[:shape,],linecolor=:black,
                            legend=false, fillalpha=0)
        
        p_plot = heatmap(xp, yp, clamp.(pin, -1, 1)'; color = :balance, size = (500,700), 
                            guidefontsize = 14, titlefont=14, top_margin = -10.0mm, bottom_margin = -10.0mm,
                            legendfont = 10, tickfont = 8, kwargs...)
        plot!(circle_shape(0,0,1),seriestype=[:shape,],linecolor=:black,
                            legend=false, fillalpha=0)

        m_plot = heatmap(xm, ym, clamp.(mi, -1, 1)'; color = :thermal, size = (500,700), 
                            guidefontsize = 14, titlefont=14, top_margin = -10.0mm, bottom_margin = -10.0mm,
                            legendfont = 10, tickfont = 8, kwargs...)
        plot!(circle_shape(0,0,1),seriestype=[:shape,],linecolor=:black,
                            legend=false, fillalpha=0)
  
        δlim = 1.0
        kwargs[:clims] = (-δlim, δlim)

        u_title = @sprintf("u at t = %.1f", t)
        v_title = @sprintf("v at t = %.1f", t)
        p_title = @sprintf("p at t = %.1f", t)
        m_title = @sprintf("m at t = %.1f", t)

        plot(u_plot, v_plot, p_plot, m_plot,
             title = [u_title v_title p_title m_title],
             layout = (2, 2))
    end
    mp4(anim, "contours_" * experiment_name * ".mp4", fps = 8)
end

function analyze_cylinder_steadystate(experiment_name)

    @info "Analyzing data from steady state cylinder flow..."
    # Analysis of coefficients is for no slip cylinder BC

    # distance function for solid cylinder
    @inline dist_cylinder(v) = sqrt((v[1])^2+(v[2])^2)-R
    # finding the sfc normal vector
    normalDist = x-> ForwardDiff.gradient(dist_cylinder,x)
    
    # function creating the projection matrix into surface normal and tangential
    function projection_matrix(N)
        if abs(N[1]) == 1 # if normal vector is entirely in the x direction
            v1 = [0; -sign(N[1]); 0]  # we want v1 = [0,0,1]
            v2 = [0; 0; -1]
        elseif abs(N[2]) == 1 || N[3]==0 # if normal vector is entirely in the y direction or 0 z comp
            v1 = [sign(N[2]); -N[1]*sign(N[2])/N[2]; 0] # we want v1 = [1,0,0]
            v1 = v1./norm(v1)
            v2 = cross(N,v1)
        else
            v1 = [sign(N[3]); 0; -N[1]*sign(N[3])/N[3]] # else we want v1 = [1, 0 , ?]
            v1 = v1./norm(v1)
            v2 = cross(N,v1)
        end
        transpose(hcat(v1,v2,N)) # creating a matrix of all 3 vectors
    end
    
    # function interpolating velocity vector at vector location xvec 
    function interp_values(xvec, Us, Vs, Ws)
        xI = xvec; # forced node is the x argument 
        uI = interpolate(Us, xI[1], xI[2], xI[3])
        vI = interpolate(Vs, xI[1], xI[2], xI[3])
        wI = interpolate(Ws, xI[1], xI[2], xI[3])
        vels = [uI; vI; wI]
        return vels
    end
    
    # function outputs normal and tangential velocity, pressure, and friction coefficient 
    # around a cylinder at M points with projection and interpolation
    function surface_values(M, U, V, W, Pnon, grid, eps, ν)
        
        # angle around circle starting due south 
        θ = LinRange(-π/2 , 3*π/2-(2*π/M), M) 
        x₀ = hcat(R*cos.(θ), R*sin.(θ), .5*ones(M,1))
        Vⁿ = zeros(M, 1)
        Vᵗ¹ = zeros(M, 1)
        p_sfc = zeros(M, 1)
        dVt = zeros(M, 1)
        
        # at each location around circle
        for m = 1:M
            
            # surface normal vector
            nm = normalDist(x₀[m,:])
            # location eps distance outside cylinder in fluid
            xI = x₀[m,:] + eps*nm
            
            # velocity vector at both points
            Vm = interp_values(x₀[m,:], U, V, W)
            Vm_ext = interp_values(xI, U, V, W)
            
            # new projected velocity vector: V_rot = [Vt1, Vt2, Vn]
            matrix = projection_matrix(nm)
            V_rot = matrix*Vm
            V_rot_ext = matrix*Vm_ext
            
            # normal and tangential velocities
            Vⁿ[m] = V_rot[3]
            Vᵗ¹[m] = V_rot[1]
            Vᵗ¹_ext = V_rot_ext[1]
            
            # surface normal derivative dVt/dn
            dVt[m] = (Vᵗ¹_ext - Vᵗ¹[m])/eps
            
            # pressure interpolated near the surface
            p_sfc[m] = interpolate(Pnon, xI[1], xI[2], xI[3])    
        end
        
        # friction coefficient
        Cf = dVt.* (2*ν)
        
        return Vⁿ,Vᵗ¹, Cf, p_sfc, θ 
    end
    
    # get data
    filepath = experiment_name * ".jld2"
    
    u_timeseries = FieldTimeSeries(filepath,  "u")
    v_timeseries = FieldTimeSeries(filepath,  "v")
    w_timeseries = FieldTimeSeries(filepath,  "w")
    p_timeseries = FieldTimeSeries(filepath,  "pNHS")
    m_timeseries = FieldTimeSeries(filepath,  "mass")

    # final iteration for steady state values
    final_iter = size(u_timeseries)[end]
    U = u_timeseries[final_iter]
    V = v_timeseries[final_iter]
    W = w_timeseries[final_iter]
    P = p_timeseries[final_iter]
    
    center_grid = p_timeseries.grid
    xm, ym, zm = nodes(m_timeseries)

    Vⁿ,Vᵗ, Cf, p_sfc, θ  = surface_values(140, U, V, W, P, center_grid, center_grid.Δx, nu)

    # function creating mesh of (x,y)     
    function meshgrid(vx::AbstractVector{T}, vy::AbstractVector{T}) where T
        m, n = length(vy), length(vx)
        vx = reshape(vx, 1, n)
        vy = reshape(vy, m, 1)
        (repeat(vx, m, 1), repeat(vy, 1, n))
    end  
    
    @inline inside_cylinder(x, y, z) = (x.^2 .+ y.^2) .<= R # immersed solid
    
    # calculates the volume integrated concentration of tracer
    function volume_integrated_concentration(x, y, mass_series)       
        X, Y = meshgrid(x, y)
        # only including concentration values outside the cylinder in the fluid
        fluid_concent = mass_series .* (1 .- inside_cylinder(X', Y', 1))
        # summing over the concentration values and multiplying by ΔxΔy at each iteration
        vol_int_concent = (sum(fluid_concent[:,:,1,:], dims = 1:2))[1,1,:] * (y[2]-y[1]) * (x[2]-x[1])
    end
     
    concentration_m = volume_integrated_concentration(xm, ym, m_timeseries)
    
    # taking the percent change in concentration at each time step in ratio to the initial amount
    percent_leakage = ((concentration_m[1] .- concentration_m) ./ concentration_m[1]) .* 100
    
    @info "Plotting Normal and Tangential Velocity..."    

    Vnt = plot(θ*180/π, Vⁿ, xlabel="θ", ylabel="Velocity", seriestype=:scatter,
        guidefontsize = 14, titlefont=14, legendfont = 8, tickfont = 8,
        xticks = [-90,0,90,180,275], xlims = (-90, 275), markersize = 3, 
        legend = true, size = (700,500), label = "Vⁿ")
    plot!(θ*180/π, Vᵗ, label= "Vᵗ", seriestype=:scatter, markersize = 3)

    @info "Plotting Pressure Coefficient..." 

    Pcoef = plot(θ*180/π, 2. .* p_sfc, xlabel="θ", ylabel= "Cp", seriestype=:scatter, 
        guidefontsize = 14, titlefont=14, legendfont = 10, tickfont = 8, 
        xticks = [-90,0,90,180,275], xlims = (-90, 275), markersize = 3, 
        legend = false, size = (700,500))

    @info "Plotting Friction Coefficient..." 

    Fcoef = plot(θ*180/π, Cf, xlabel="θ", ylabel= "Cf", seriestype=:scatter, 
        guidefontsize = 14, titlefont=14,legendfont = 10, tickfont = 8, 
        xticks = [-90,0,90,180,275], xlims = (-90, 275), markersize = 3, 
        legend = false, size = (700,500))
    
    @info "Plotting Concentration Leakage"
    mConcent = plot(m_timeseries.times, percent_leakage, xlabel="t", ylabel= "% Leakage", 
        seriestype=:scatter, guidefontsize = 14, titlefont=14, legendfont = 10, 
        tickfont = 8, markersize = 3, legend = false, size = (700,500))
    
    Vnt_title = @sprintf("||Vⁿ||₂  = %.3f, ||Vᵗ||₂  = %.3f", norm(Vⁿ,2), norm(Vᵗ,2))
    psfc_title = "Pressure Coefficient"
    cf_title = "Friction Coefficient"
    trac_title = "% Concentration Leakage"
    
    analysisplot = plot(Vnt, Pcoef, Fcoef, mConcent,
             title = [Vnt_title psfc_title cf_title trac_title],
             layout = (2, 2), left_margin=5mm, bottom_margin=10mm)

    Plots.savefig(analysisplot, "analysis_" * experiment_name * ".png")
end

advection = CenteredSecondOrder()
experiment_name = run_cylinder_steadystate(Nh = 350, advection = advection, radius = R, stop_time = 100, ν = nu)
visualize_cylinder_steadystate(experiment_name)
analyze_cylinder_steadystate(experiment_name)


