using Oceananigans
using GLMakie

using Oceananigans.BoundaryConditions: fill_halo_regions!, ImpenetrableBoundaryCondition
using Oceananigans.Solvers: ConjugateGradientPoissonSolver, DiagonallyDominantPreconditioner

using Oceananigans.Operators: ∇²ᶜᶜᶜ
using Oceananigans.Solvers: ConjugateGradientSolver, solve!
using Oceananigans.Architectures: architecture
using Oceananigans.Utils: launch!
using Oceananigans.Units

using KernelAbstractions

@kernel function streamfunction_laplacian!(∇²ψ, grid, ψ)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²ψ[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, ψ)
end

function compute_streamfunction_laplacian!(∇²ψ, ψ)
    grid = ψ.grid
    arch = architecture(grid)
    fill_halo_regions!(ψ)
    launch!(arch, grid, :xyz, streamfunction_laplacian!, ∇²ψ, grid, ψ)
    return nothing
end

filename = "surface_wave_induced_flow.jld2"
ut = FieldTimeSeries(filename, "u")
wt = FieldTimeSeries(filename, "w")

grid = ut.grid

Nx, Ny, Nz = size(grid)
Nx⁻ = Nx ÷ 2

left_grid = RectilinearGrid(size = (Nx⁻, Nz),
                            x = (-4kilometers, 4kilometers),
                            z = (-4096, 0),
                            topology = (Periodic, Flat, Bounded))

right_grid = RectilinearGrid(size = (Nx⁻, Nz),
                             x = (4kilometers, 12kilometers),
                             z = (-4096, 0),
                             topology = (Periodic, Flat, Bounded))

uL = XFaceField(left_grid)
wL = ZFaceField(left_grid)

uR = XFaceField(right_grid)
wR = ZFaceField(right_grid)

Nt = length(ut)
un = ut[Nt]
wn = wt[Nt]

interior(uL) .= interior(un, 1:Nx⁻, :, :)
interior(uR) .= interior(un, Nx⁻+1:Nx, :, :)

interior(wL) .= interior(wn, 1:Nx⁻, :, :)
interior(wR) .= interior(wn, Nx⁻+1:Nx, :, :)

uR_s = XFaceField(right_grid)
uR_d = XFaceField(right_grid)
wR_s = ZFaceField(right_grid)
wR_d = ZFaceField(right_grid)

ϵ = 0.1
λ = 60 # meters
k = 2π / λ
g = 9.81
c = sqrt(g / k)
δ = 400
cᵍ = c / 2
Uˢ = ϵ^2 * c
A(ξ) = exp(- ξ^2 / 2δ^2)
ûˢ(z) = Uˢ * exp(2k * z)
uˢ(x, z, t) = A(x - cᵍ * t) * ûˢ(z)
tn = ut.times[Nt]
set!(uR_s, (x, z) -> uˢ(x, z, tn))
interior(uR_d) .= interior(uR) .- interior(uR_s)

uL_d = XFaceField(left_grid)
uL_s = XFaceField(left_grid)
wL_d = ZFaceField(left_grid)
wL_s = ZFaceField(left_grid)

interior(uL_d) .= -interior(uR_d)
interior(uL_s) .= interior(uL) .+ interior(uR_d)
interior(wL_d) .= -interior(wR_d)
interior(wL_s) .= interior(wL) .+ interior(wR_d)

for f in (uL, uL_s, uL_d, uR, uR_s, uR_d, wL, wL_s, wL_d, wR, wR_s, wR_d)
    fill_halo_regions!(f)
end

r = Field(∂z(uL) - ∂x(wL))
compute!(r)

cg_solver = ConjugateGradientSolver(compute_streamfunction_laplacian!;
                                    reltol = 1e-8,
                                    abstol = 0,
                                    preconditioner = DiagonallyDominantPreconditioner(),
                                    template_field = r)

ψ_bcs = FieldBoundaryConditions(left_grid, (Face, Face, Center),
                                top=ImpenetrableBoundaryCondition(),
                                bottom=ImpenetrableBoundaryCondition())

ψL   = Field{Face, Center, Face}(left_grid, boundary_conditions=ψ_bcs)
ψL_s = Field{Face, Center, Face}(left_grid, boundary_conditions=ψ_bcs)
ψL_d = Field{Face, Center, Face}(left_grid, boundary_conditions=ψ_bcs)
ψR   = Field{Face, Center, Face}(right_grid, boundary_conditions=ψ_bcs)
#ψR_s = Field{Face, Center, Face}(right_grid, boundary_conditions=ψ_bcs)
ψR_d = Field{Face, Center, Face}(right_grid, boundary_conditions=ψ_bcs)

ψR_s = Field(CumulativeIntegral(uR_s; dims=3))
compute!(ψR_s)
# Ψ = mean(ψR_s)
# interior(ψR_s) .-= Ψ

rR = Field(∂x(wR) - ∂z(uR))
compute!(rR)
solve!(ψR, cg_solver, rR)
# Ψ = mean(ψR)
# interior(ψR) .-= Ψ

# u = - ∂z(ψ)
# w = + ∂x(ψ)
# ⟹  ∇²ψ = w_x - u_z
rL = Field(∂x(wL) - ∂z(uL))
compute!(rL)
solve!(ψL, cg_solver, rL)
# Ψ = mean(ψL)
# interior(ψL) .-= Ψ

# Return flow
interior(ψR_d) .= interior(ψR) .- interior(ψR_s)

# Return flow
interior(ψL_d) .= - interior(ψR_d)

rL_s = Field(∂x(wL_s) - ∂z(uL_s))
compute!(rL_s)
solve!(ψL_s, cg_solver, rL_s)
# interior(ψL_s) .= interior(ψL) - interior(ψR_d)

#=
us = (uL, uL_s, uL_d, uR, uR_s, uR_d)
ws = (wL, wL_s, wL_d, wR, wR_s, wR_d)
ψs = (ψL, ψL_s, ψL_d, ψR, ψR_s, ψR_d)

using Statistics

for ψ in ψs
    Ψ = mean(ψ)
    interior(ψ) .-= Ψ
end
=#

fig = Figure(size=(800, 800))

axLt = Axis(fig[1, 1], xlabel="x (m)", ylabel="z (m)", title="Total wave-transmitted x-velocity")
axRt = Axis(fig[1, 2], xlabel="x (m)", ylabel="z (m)", title="Total wave-coincident flow")

axLd = Axis(fig[2, 1], xlabel="x (m)", ylabel="z (m)", title="Wave-transmitted momentum-containing component")
axRd = Axis(fig[2, 2], xlabel="x (m)", ylabel="z (m)", title="Wave-coindicent far-field component")

axLs = Axis(fig[3, 1], xlabel="x (m)", ylabel="z (m)", title="Wave-transmitted adjument component")
axRs = Axis(fig[3, 2], xlabel="x (m)", ylabel="z (m)", title="Wave-coincident near-field component")

Ψ⁺ = 1e-1
Ψ⁻ = 1e-2
dΨ = 1e-3
levels = vcat(-Ψ⁺:dΨ:-Ψ⁻, Ψ⁺:dΨ:Ψ⁻) 
kw = (; levels)
Nz = size(uL, 3)
x, y, z = nodes(ψL)
contour!(axLt, x, z, interior(ψL, :, 1, :);   levels=-1e-2:1e-4:1e-2)
contour!(axLd, x, z, interior(ψL_d, :, 1, :); levels=20) #levels=-1e-2:1e-4:1e-2)
contour!(axLs, x, z, interior(ψL_s, :, 1, :); levels=20) #levels=-1e-2:1e-4:1e-2)

# x, y, z = nodes(uR)
contour!(axRt, x, z, interior(ψR, :, 1, :); levels=15)
contour!(axRd, x, z, interior(ψR_d, :, 1, :); levels=20) #levels=-1e-2:1e-4:1e-2)
contour!(axRs, x, z, interior(ψR_s, :, 1, :); levels=20)

for ax in (axLt,
           axRt,
           axLs,
           axRs,
           axLd,
           axRd)

    x₀ = 2kilometers
    hidedecorations!(ax)
    ylims!(ax, -1500, 0)
    xlims!(ax, -x₀, x₀)
end

display(fig)

