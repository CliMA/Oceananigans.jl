using
    Plots,
    #PyPlot,
    FFTW,
    Oceananigans

function make_temperature_movie(model::Model, fw::NetCDFOutputWriter)
    n_frames = Int(model.clock.iteration / fw.output_frequency)

    xC, yC, zC = model.grid.xC, model.grid.yC, model.grid.zC

    print("Creating temperature movie... ($n_frames frames)\n")

    Plots.gr()
    default(dpi=300)
    movie = @animate for tidx in 0:n_frames
        print("\rframe = $tidx / $n_frames   ")
        temperature = read_output(fw, "T", tidx*fw.output_frequency)
        Plots.heatmap(xC, zC, rotl90(temperature[:, Int(ceil(model.grid.Ny/2)), :]) .- 283, color=:balance,
                      clims=(-0.05, 0.05), aspect_ratio=:equal,
                      title="T @ t=$(tidx*fw.output_frequency)")
    end

    mp4(movie, "free_convection_$(round(Int, time())).mp4", fps=30)
end

function plot_Nusselt_number_diagnostics(model::Model, Nu_wT_diag::Nusselt_wT, Nu_Chi_diag::Nusselt_Chi)
    println("Plotting Nusselt number diagnostics...")

    Nx, Ny, Nz = model.grid.Nx, model.grid.Ny, model.grid.Nz
    t = 0:model.clock.Δt:model.clock.time

    PyPlot.plot(t, Nu_wT_diag.Nu, label="Nu_wT")
    PyPlot.plot(t, Nu_wT_diag.Nu_inst, label="Nu_wT_inst")
    PyPlot.plot(t, Nu_Chi_diag.Nu, label="Nu_Chi")
    PyPlot.plot(t, Nu_Chi_diag.Nu_inst, label="Nu_Chi_inst")

    PyPlot.title("Rayleigh–Bénard convection ($Nx×$Ny×$Nz) @ Ra=5000")
    PyPlot.xlabel("Time (s)")
    PyPlot.ylabel("Nusselt number Nu")
    PyPlot.legend()
    PyPlot.savefig("free_convection_nusselt_diag.png", dpi=300, format="png", transparent=false)
end

Nx, Ny, Nz = 512, 1, 256
Lx, Ly, Lz = 500, 500, 500
Nt, Δt = 40*100, 10
ν, κ = 1e-2, 1e-2

model = Model(N=(Nx, Ny, Nz), L=(Lx, Ly, Lz), ν=ν, κ=κ)
model.boundary_conditions = BoundaryConditions(:periodic, :periodic, :rigid_lid, :no_slip)
#
@inline fT(u,v,w,T,S,Nx,Ny,Nz,dx,dy,dz,i,j,k ) = ifelse(k ==1, -9e-6 + 1e-6*sin.( 2*pi * i / 512),0)
model.forcing = Forcing(nothing,nothing,nothing, fT, nothing)

# Write temperature field to disk every 10 time steps.
output_writer = NetCDFOutputWriter(dir=".", prefix="convection", frequency=40)
push!(model.output_writers, output_writer)

# Time stepping
time_step!(model, Nt, Δt)

make_temperature_movie(model, output_writer)
