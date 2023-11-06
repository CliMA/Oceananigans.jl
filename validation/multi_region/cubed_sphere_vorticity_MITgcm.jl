#=
Download the directory MITgcm_Output from  
https://www.dropbox.com/scl/fo/qr024ly4t3eq38jsi0sdj/h?rlkey=zbq50ud1mtv8l05wxjarulpr3&dl=0
and place it in the path validation/multi_region/. Then run this script from the same path as
include("cubed_sphere_vorticity_MITgcm.jl")
=#

using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Operators
using Oceananigans.Utils: Iterate
using CairoMakie

Nx = 32
Ny = 32
Nz = 1

Lz = 1
R = 6370e3 # sphere's radius
U = 1 # velocity scale

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-Lz, 0),
                                  radius = R,
                                  horizontal_direction_halo = 2,
                                  partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

# Solid body rotation
omegaprime = 38.60328935834681/R
PI = 3.14159265358979323844
Omega = 2PI/86400

ψᵣ(λ, φ, z) = -R^2*omegaprime/(2Omega)*2Omega*sind(φ)

# for φʳ = 90; ψᵣ(λ, φ, z) = - U * R * sind(φ)
#              uᵣ(λ, φ, z) = - 1 / R * ∂φ(ψᵣ) = U * cosd(φ)
#              vᵣ(λ, φ, z) = + 1 / (R * cosd(φ)) * ∂λ(ψᵣ) = 0
#              ζᵣ(λ, φ, z) = - 1 / (R * cosd(φ)) * ∂φ(uᵣ * cosd(φ)) = 2 * (U / R) * sind(φ)

ψ = Field{Face, Face, Center}(grid)

# set fills only interior points; to compute u and v we need information in the halo regions
set!(ψ, ψᵣ)

# Note: fill_halo_regions! works for (Face, Face, Center) field, *except* for the
# two corner points that do not correspond to an interior point!
# We need to manually fill the Face-Face halo points of the two corners
# that do not have a corresponding interior point.
for region in [1, 3, 5]
    i = 1
    j = Ny+1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

for region in [2, 4, 6]
    i = Nx+1
    j = 1
    for k in 1:Nz
        λ = λnode(i, j, k, grid[region], Face(), Face(), Center())
        φ = φnode(i, j, k, grid[region], Face(), Face(), Center())
        ψ[region][i, j, k] = ψᵣ(λ, φ, 0)
    end
end

for passes in 1:3
    fill_halo_regions!(ψ)
end

u = XFaceField(grid)
v = YFaceField(grid)

ut = XFaceField(grid)
vt = YFaceField(grid)

function create_test_data(grid, region; trailing_zeros=0)
    Nx, Ny, Nz = size(grid)
    (Nx > 9 || Ny > 9) && error("you provided (Nx, Ny) = ($Nx, $Ny); use a grid with Nx, Ny ≤ 9.")
    !(trailing_zeros isa Integer) && error("trailing_zeros has to be an integer")
    factor = 10^trailing_zeros
    return factor .* [100region + 10i + j for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

if Nx ≤ 9
    region = Iterate(1:6)
    @apply_regionally u_data = create_test_data(grid, region, trailing_zeros=0)
    @apply_regionally v_data = create_test_data(grid, region, trailing_zeros=1)
    set!(ut, u_data)
    set!(vt, v_data)
end

# What we want eventually:
# u .= - ∂y(ψ)
# v .= + ∂x(ψ)

# for region in 1:number_of_regions(grid)
#     u[region] .= - ∂y(ψ[region])
#     v[region] .= + ∂x(ψ[region])
# end

for region in 1:number_of_regions(grid)
    for j in 1:grid.Ny, i in 1:grid.Nx, k in 1:grid.Nz
        u[region][i, j, k] = - (ψ[region][i, j+1, k] - ψ[region][i, j, k]) / grid[region].Δyᶠᶜᵃ[i, j]
        v[region][i, j, k] =   (ψ[region][i+1, j, k] - ψ[region][i, j, k]) / grid[region].Δxᶜᶠᵃ[i, j]
    end
end

function fill_velocity_halos!(u, v)
    for passes in 1:3
        fill_halo_regions!(u)
        fill_halo_regions!(v)
        @apply_regionally replace_horizontal_vector_halos!((; u, v, w = nothing), grid)
    end

    for region in [1, 3, 5]
        region_south = mod(region + 4, 6) + 1
        region_east = region + 1
        region_north = mod(region + 2, 6)
        region_west = mod(region + 4, 6)

        # Northwest corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(-u[region_west][2, Ny-Hy+1:Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-v[region_west][1, Ny-Hy+2:Ny, k]')
            # Local x direction
            u[region][1-Hx:0, Ny+1, k] .= reverse(-u[region_north][2:Hx+1, Ny, k])
            v[region][1-Hx:0, Ny+1, k] .= -u[region_west][1, Ny-Hx+1:Ny, k]
        end

        # Northeast corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= -v[region_north][1:Hy, 1, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1:Hy, Ny, k]'
            # Local x direction
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= u[region_north][1:Hx, 1, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][1:Hy, 1, k]
        end

        # Southwest corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][0, 1-Hy:0, k] .= u[region_west][Nx, Ny-Hy+1:Ny, k]'
            v[region][0, 1-Hy:0, k] .= v[region_west][Nx, Ny-Hy+1:Ny, k]'
            # Local x direction
            u[region][1-Hx:0, 0, k] .= v[region_south][1, Ny-Hx+1:Ny, k]
            v[region][1-Hx:0, 0, k] .= -u[region_south][2, Ny-Hx+1:Ny, k]
        end

        # Southeast corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][Nx+1, 1-Hy:0, k] .= reverse(v[region_east][1:Hy, 1, k]')
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-u[region_east][2:Hy+1, 1, k]')
            # Local x direction
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx, Ny-Hx+2:Ny, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= u[region_south][Nx, Ny-Hx+1:Ny, k]
        end
    end
    
    for region in [2, 4, 6]
        region_south = mod(region + 3, 6) + 1
        region_east = mod(region, 6) + 2
        region_north = mod(region, 6) + 1
        region_west = region - 1

        # Northwest corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][0, Ny+1:Ny+Hy, k] .= reverse(v[region_west][Nx-Hy+1:Nx, Ny, k]')
            v[region][0, Ny+1, k] = -u[region][1, Ny, k]
            v[region][0, Ny+2:Ny+Hy, k] .= reverse(-u[region_west][Nx-Hy+2:Nx, Ny, k]')
            # Local x direction
            u[region][1-Hx:0, Ny+1, k] .= reverse(-v[region_north][1, 2:Hx+1, k])
            v[region][1-Hx:0, Ny+1, k] .= reverse(u[region_north][1, 1:Hx, k])
        end

        # Northeast corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][Nx+1, Ny+1:Ny+Hy, k] .= u[region_east][1, 1:Hy, k]'
            v[region][Nx+1, Ny+1:Ny+Hy, k] .= v[region_east][1, 1:Hy, k]'
            # Local x direction
            u[region][Nx+1:Nx+Hx, Ny+1, k] .= u[region_east][1:Hx, 1, k]
            v[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_east][1:Hx, 1, k]
        end
        
        # Southwest corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][0, 1-Hy:0, k] .= -v[region_west][Nx-Hy+1:Nx, 2, k]'
            v[region][0, 1-Hy:0, k] .= u[region_west][Nx-Hy+1:Nx, 1, k]'
            # Local x direction
            u[region][1-Hx:0, 0, k] .= u[region_south][Nx-Hx+1:Nx, Ny, k]
            v[region][1-Hx:0, 0, k] .= v[region_south][Nx-Hx+1:Nx, Ny, k]
        end
        
        # Southeast corner
        for k in -Hz+1:Nz+Hz
            # Local y direction
            u[region][Nx+1, 1-Hy:0, k] .= -v[region_south][Nx-Hy+1:Nx, 1, k]'
            v[region][Nx+1, 1-Hy:0, k] .= reverse(-v[region_east][Nx, 2:Hy+1, k]')
            # Local x direction
            u[region][Nx+1, 0, k] = -v[region][Nx, 1, k]
            u[region][Nx+2:Nx+Hx, 0, k] .= reverse(-u[region_south][Nx-Hx+2:Nx, 1, k])
            v[region][Nx+1:Nx+Hx, 0, k] .= reverse(-v[region_south][Nx-Hx+1:Nx, 2, k])
        end        
    end

    return nothing
end

fill_velocity_halos!(u, v)
if Nx ≤ 9
    fill_velocity_halos!(ut, vt)
end

# Now compute vorticity
using Oceananigans.Utils
using KernelAbstractions: @kernel, @index

ζ = Field{Face, Face, Center}(grid)

@kernel function _compute_vorticity!(ζ, grid, u, v)
    i, j, k = @index(Global, NTuple)
    @inbounds ζ[i, j, k] = ζ₃ᶠᶠᶜ(i, j, k, grid, u, v)
end

offset = -1 .* halo_size(grid)
@apply_regionally begin
    params = KernelParameters(total_size(ζ[1]), offset)
    launch!(CPU(), grid, params, _compute_vorticity!, ζ, grid, u, v)
end

nan = convert(eltype(grid), NaN)

for region in 1:number_of_regions(grid)
    #=
    u[region][1-Hx:0, :, :] .= nan
    u[region][Nx+2:Nx+Hx, :, :] .= nan
    u[region][:, 1-Hy:0, :] .= nan
    u[region][:, Ny+1:Ny+Hy, :] .= nan
    v[region][1-Hx:0, :, :] .= nan
    v[region][Nx+1:Nx+Hx, :, :] .= nan
    v[region][:, 1-Hy:0, :] .= nan
    v[region][:, Ny+2:Ny+Hy, :] .= nan
    =#
    ζ[region][1-Hx:0, :, :] .= nan
    ζ[region][Nx+2:Nx+Hx, :, :] .= nan
    ζ[region][:, 1-Hy:0, :] .= nan
    ζ[region][:, Ny+2:Ny+Hy, :] .= nan
end

function recompute_vorticity_corners_using_interior_points!(ζ)
    Nx, Ny, Nz = size(ζ.grid)
    Hx, Hy, Hz = halo_size(ζ.grid)
    for region in [1, 3, 5]

        region_south = mod(region + 4, 6) + 1
        region_east = region + 1
        region_north = mod(region + 2, 6)
        region_west = mod(region + 4, 6)
        
        # Northwest corner
        i = 1; j = Ny + 1
        
        # Indices of interior points
        i₁ = 1; j₁ = Ny
        i₂ = 1; j₂ = Ny
        i₃ = 1; j₃ = Ny
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δx_qᶠᶜᶜ(i₁, j₁, k, grid[region], u[region])
                                  + Δx_qᶠᶜᶜ(i₂, j₂, k, grid[region_north], u[region_north]) 
                                  + Δx_qᶠᶜᶜ(i₃, j₃, k, grid[region_west], u[region_west])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
        # Northeast corner
        i = Nx + 1; j = Ny + 1
        
        # Indices of interior points
        i₁ = 1; j₁ = Ny
        i₂ = 1; j₂ = 1
        i₃ = 1; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δx_qᶠᶜᶜ(i₁, j₁, k, grid[region_east], u[region_east]) 
                                  + Δy_qᶜᶠᶜ(i₂, j₂, k, grid[region_north], v[region_north])
                                  - Δx_qᶠᶜᶜ(i₃, j₃, k, grid[region_north], u[region_north])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
        # Southwest corner
        i = 1; j = 1

        # Indices of interior points
        i₁ = 1; j₁ = Ny
        i₂ = 1; j₂ = 1
        i₃ = 1; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δx_qᶠᶜᶜ(i₁, j₁, k, grid[region_south], u[region_south])
                                  + Δy_qᶜᶠᶜ(i₂, j₂, k, grid[region], v[region])
                                  - Δx_qᶠᶜᶜ(i₃, j₃, k, grid[region], u[region])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
        # Southeast corner
        i = Nx + 1; j = 1
        
        # Indices of interior points
        i₁ = 1; j₁ = 1
        i₂ = 1; j₂ = 1
        i₃ = Nx; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δy_qᶜᶠᶜ(i₁, j₁, k, grid[region_east], v[region_east])
                                  - Δx_qᶠᶜᶜ(i₂, j₂, k, grid[region_east], u[region_east])
                                  - Δy_qᶜᶠᶜ(i₃, j₃, k, grid[region], v[region])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
    end

    for region in [2, 4, 6]

        region_south = mod(region + 3, 6) + 1
        region_east = mod(region, 6) + 2
        region_north = mod(region, 6) + 1
        region_west = region - 1
        
        # Northwest corner
        i = 1; j = Ny + 1
        
        # Indices of interior points
        i₁ = 1; j₁ = Ny
        i₂ = 1; j₂ = 1
        i₃ = 1; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δx_qᶠᶜᶜ(i₁, j₁, k, grid[region], u[region])
                                  + Δy_qᶜᶠᶜ(i₂, j₂, k, grid[region_north], v[region_north]) 
                                  - Δx_qᶠᶜᶜ(i₃, j₃, k, grid[region_north], u[region_north])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
        # Northeast corner
        i = Nx + 1; j = Ny + 1

        # Indices of interior points
        i₁ = 1; j₁ = 1
        i₂ = 1; j₂ = 1
        i₃ = Nx; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (+ Δy_qᶜᶠᶜ(i₁, j₁, k, grid[region_east], v[region_east]) 
                                  - Δx_qᶠᶜᶜ(i₂, j₂, k, grid[region_east], u[region_east])
                                  - Δy_qᶜᶠᶜ(i₃, j₃, k, grid[region_north], v[region_north])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end    
        
        # Southwest corner
        i = 1; j = 1

        # Indices of interior points
        i₁ = Nx; j₁ = 1
        i₂ = 1; j₂ = 1
        i₃ = 1; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (- Δy_qᶜᶠᶜ(i₁, j₁, k, grid[region_west], v[region_west])
                                  + Δy_qᶜᶠᶜ(i₂, j₂, k, grid[region], v[region])
                                  - Δx_qᶠᶜᶜ(i₃, j₃, k, grid[region], u[region])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end
        
        # Southeast corner
        i = Nx + 1; j = 1
        
        # Indices of interior points
        i₁ = Nx; j₁ = 1
        i₂ = Nx; j₂ = 1
        i₃ = Nx; j₃ = 1
        
        for k in -Hz+1:Nz+Hz
            ζ[region][i, j, k] = (- Δy_qᶜᶠᶜ(i₁, j₁, k, grid[region_south], v[region_south])
                                  - Δy_qᶜᶠᶜ(i₂, j₂, k, grid[region_east], v[region_east])
                                  - Δy_qᶜᶠᶜ(i₃, j₃, k, grid[region], v[region])) / Azᶠᶠᶜ(i, j, k, grid[region]) * 4/3
        end

    end

    return nothing
end

f = Field{Face, Face, Center}(grid)
for region in 1:number_of_regions(grid)
    f[region] .= 2 * Omega * sind.(grid[region].φᶠᶠᵃ)
end

function panel_wise_visualization(field, k=1; hide_decorations = true, colorrange = (-1, 1), colormap = :balance)

    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, parent(getregion(field, 1).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, parent(getregion(field, 2).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, parent(getregion(field, 3).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, parent(getregion(field, 4).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, parent(getregion(field, 5).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, parent(getregion(field, 6).data[:, :, k]); colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    if hide_decorations
        hidedecorations!(ax_1)
        hidedecorations!(ax_2)
        hidedecorations!(ax_3)
        hidedecorations!(ax_4)
        hidedecorations!(ax_5)
        hidedecorations!(ax_6)
    end

    return fig
end

u_theoretical = XFaceField(grid)
v_theoretical = YFaceField(grid)

for region in 1:number_of_regions(grid)
    u_theoretical[region][1:Nx, 1:Ny, :] .= U * cosd.(grid[region].φᶠᶜᵃ[1:Nx, 1:Ny, :])
    v_theoretical[region].data .= 0
end

function panel_wise_visualization_MITgcm(x, y, field; hide_decorations = true, colorrange = (-1, 1), colormap = :balance)
    
    fig = Figure(resolution = (2450, 1400))

    axis_kwargs = (xlabelsize = 22.5, ylabelsize = 22.5, xticklabelsize = 17.5, yticklabelsize = 17.5, aspect = 1.0, 
                   xlabelpadding = 10, ylabelpadding = 10, titlesize = 27.5, titlegap = 15, titlefont = :bold,
                   xlabel = "Local x direction", ylabel = "Local y direction")

    ax_1 = Axis(fig[3, 1]; title = "Panel 1", axis_kwargs...)
    hm_1 = heatmap!(ax_1, x[:, :, 1], y[:, :, 1], field[:, :, 1]; colorrange, colormap)
    Colorbar(fig[3, 2], hm_1)

    ax_2 = Axis(fig[3, 3]; title = "Panel 2", axis_kwargs...)
    hm_2 = heatmap!(ax_2, x[:, :, 2], y[:, :, 2], field[:, :, 2]; colorrange, colormap)
    Colorbar(fig[3, 4], hm_2)

    ax_3 = Axis(fig[2, 3]; title = "Panel 3", axis_kwargs...)
    hm_3 = heatmap!(ax_3, x[:, :, 3], y[:, :, 3], field[:, :, 3]; colorrange, colormap)
    Colorbar(fig[2, 4], hm_3)

    ax_4 = Axis(fig[2, 5]; title = "Panel 4", axis_kwargs...)
    hm_4 = heatmap!(ax_4, x[:, :, 4], y[:, :, 4], field[:, :, 4]; colorrange, colormap)
    Colorbar(fig[2, 6], hm_4)

    ax_5 = Axis(fig[1, 5]; title = "Panel 5", axis_kwargs...)
    hm_5 = heatmap!(ax_5, x[:, :, 5], y[:, :, 5], field[:, :, 5]; colorrange, colormap)
    Colorbar(fig[1, 6], hm_5)

    ax_6 = Axis(fig[1, 7]; title = "Panel 6", axis_kwargs...)
    hm_6 = heatmap!(ax_6, x[:, :, 6], y[:, :, 6], field[:, :, 6]; colorrange, colormap)
    Colorbar(fig[1, 8], hm_6)

    if hide_decorations
        hidedecorations!(ax_1)
        hidedecorations!(ax_2)
        hidedecorations!(ax_3)
        hidedecorations!(ax_4)
        hidedecorations!(ax_5)
        hidedecorations!(ax_6)
    end

    return fig
    
end

function read_big_endian_coordinates(filename)
    # Open the file in binary read mode
    open(filename, "r") do io
        # Calculate the number of Float64 values in the file
        n = filesize(io) ÷ sizeof(Float64)
        
        # Ensure n = 32x32
        if n != 32 * 32
            error("File size does not match the expected size for one 32x32 field")
        end

        # Initialize an array to hold the data
        data = Vector{Float64}(undef, n)

        # Read the data into the array
        read!(io, data)

        # Convert from big-endian to native endianness
        native_data = reshape( bswap.(data), 32, 32) 
        
        return native_data
    end
end

function read_big_endian_diagnostic_data(filename)
    # Open the file in binary read mode
    open(filename, "r") do io
        # Calculate the number of Float64 values in the file
        n = filesize(io) ÷ sizeof(Float64)

        # Ensure n = 2x32x32
        if n != 2 * 32 * 32
            error("File size does not match the expected size for two 32x32 fields")
        end

        # Initialize an array to hold the data
        data = Vector{Float64}(undef, n)

        # Read the data into the array
        read!(io, data)

        # Convert from big-endian to native endianness
        native_data = bswap.(data)

        # Extract and reshape the data to form two 32x32 fields
        momKE = reshape(native_data[1:32*32], 32, 32)
        momVort3 = reshape(native_data[32*32+1:end], 32, 32)

        return momKE, momVort3
    end
end

Nx = 32; Ny = 32
XGs = zeros(Nx, Ny, 6)
YGs = zeros(Ny, Ny, 6)
Us = zeros(Nx, Ny, 6)
Vs = zeros(Nx, Ny, 6)
momVort3s = zeros(Nx, Ny, 6)
Az_ffcs = zeros(Nx, Ny, 6)

panel_indices = [1, 2, 3, 4, 5, 6]

for (iter, pidx) in enumerate(panel_indices)
    XG = read_big_endian_coordinates("MITgcm_Output/2023-11-06/XG.00$(pidx).001.data")
    YG = read_big_endian_coordinates("MITgcm_Output/2023-11-06/YG.00$(pidx).001.data")
    U = read_big_endian_coordinates("MITgcm_Output/2023-11-06/U.0000000000.00$(pidx).001.data")
    V = read_big_endian_coordinates("MITgcm_Output/2023-11-06/V.0000000000.00$(pidx).001.data")
    Az_ffc = read_big_endian_coordinates("MITgcm_Output/2023-11-06/RAZ.00$(pidx).001.data")
    momKE, momVort3 = read_big_endian_diagnostic_data("MITgcm_Output/2023-11-06/momDiag.0000000000.00$(pidx).001.data")
    XGs[:, :, iter] = XG
    YGs[:, :, iter] = YG
    Us[:, :, iter] = U
    Vs[:, :, iter] = V
    momVort3s[:, :, iter] = momVort3
    Az_ffcs[:, :, iter] = Az_ffc
end

# at the poles, the longitudes are ill-defined;
# we ensure both grids have the same values of longitude
# at the poles before we compare them
XGs[YGs .== +90] .= grid[3].λᶠᶠᵃ[grid[3].φᶠᶠᵃ .== +90]
XGs[YGs .== -90] .= grid[6].λᶠᶠᵃ[grid[6].φᶠᶠᵃ .== -90]

for region in 1:6
    @show grid[region].λᶠᶠᵃ[1:32, 1:32] ≈ XGs[:, :, region]
    @show grid[region].φᶠᶠᵃ[1:32, 1:32] ≈ YGs[:, :, region]
end

ζ_w_interior = deepcopy(ζ)
recompute_vorticity_corners_using_interior_points!(ζ_w_interior)

for region in 1:6
    @show region, ζ_w_interior[region] ≈ ζ[region]
end

ψ_Array = zeros(Nx+1, Ny+1, 6)
u_Array = zeros(Nx, Ny, 6)
v_Array = zeros(Nx, Ny, 6)
ζ_Array = zeros(Nx+1, Ny+1, 6)
f_Array = zeros(Nx+1, Ny+1, 6)
for pidx in 1:6
    ψ_Array[1:Nx+1, 1:Ny+1, pidx] = getregion(ψ, pidx).data[1:Nx+1, 1:Ny+1, 1]
    u_Array[1:Nx, 1:Ny, pidx] = getregion(u, pidx).data[1:Nx, 1:Ny, 1]
    v_Array[1:Nx, 1:Ny, pidx] = getregion(v, pidx).data[1:Nx, 1:Ny, 1]
    ζ_Array[1:Nx+1, 1:Ny+1, pidx] = getregion(ζ, pidx).data[1:Nx+1, 1:Ny+1, 1]
    f_Array[1:Nx+1, 1:Ny+1, pidx] = getregion(f, pidx).data[1:Nx+1, 1:Ny+1, 1]
end

u_difference = u_Array - Us
v_difference = v_Array - Vs
momVort3s_difference = ζ_Array[1:Nx, 1:Ny, :] - momVort3s[:, :, :]

useSymmetricColorRange = true
if useSymmetricColorRange
    ψ_Array_limit = max(maximum(abs.(ψ_Array)), minimum(abs.(ψ_Array)))
    ψ_Array_color_range = (-ψ_Array_limit, ψ_Array_limit)
    u_Array_limit = max(maximum(abs.(u_Array)), minimum(abs.(u_Array)))
    u_Array_color_range = (-u_Array_limit, u_Array_limit)
    u_difference_limit = max(maximum(abs.(u_difference)), minimum(abs.(u_difference)))
    u_difference_color_range = (-u_difference_limit, u_difference_limit)
    v_Array_limit = max(maximum(abs.(v_Array)), minimum(abs.(v_Array)))
    v_Array_color_range = (-v_Array_limit, v_Array_limit)
    v_difference_limit = max(maximum(abs.(v_difference)), minimum(abs.(v_difference)))
    v_difference_color_range = (-v_difference_limit, v_difference_limit)
    ζ_Array_limit = max(maximum(abs.(ζ_Array)), minimum(abs.(ζ_Array)))
    ζ_Array_color_range = (-ζ_Array_limit, ζ_Array_limit)
    f_Array_limit = max(maximum(abs.(f_Array)), minimum(abs.(f_Array)))
    f_Array_color_range = (-f_Array_limit, f_Array_limit)
    momVort3s_limit = max(maximum(abs.(momVort3s)), minimum(abs.(momVort3s)))
    momVort3s_color_range = (-momVort3s_limit, momVort3s_limit)
    momVort3s_limit = max(maximum(abs.(momVort3s_difference)), minimum(abs.(momVort3s_difference)))
    momVort3s_difference_color_range = (-momVort3s_limit, momVort3s_limit)
else
    ψ_Array_color_range = (minimum(ψ_Array), maximum(ψ_Array))
    u_Array_color_range = (minimum(u_Array), maximum(u_Array))
    u_difference_color_range = (minimum(u_difference), maximum(u_difference))
    v_Array_color_range = (minimum(v_Array), maximum(v_Array))
    v_difference_color_range = (minimum(v_difference), maximum(v_difference))
    ζ_Array_color_range = (minimum(ζ_Array), maximum(ζ_Array))
    f_Array_color_range = (minimum(f_Array), maximum(f_Array))
    momVort3s_color_range = (minimum(momVort3s), maximum(momVort3s))
    momVort3s_difference_color_range = (minimum(momVort3s_difference), maximum(momVort3s_difference))
end

# Plot the streamfunction ψ.
fig = panel_wise_visualization(ψ, colorrange=ψ_Array_color_range)
save("streamfunction.png", fig)

# Plot the zonal velocity u.
fig = panel_wise_visualization(u, colorrange=u_Array_color_range)
save("zonal_velocity.png", fig)

# Plot the meridional velocity v.
fig = panel_wise_visualization(v, colorrange=v_Array_color_range)
save("meridional_velocity.png", fig)

# Plot the Coriolis parameter.
fig = panel_wise_visualization(f, colorrange=f_Array_color_range)
save("f.png", fig)

# Plot the vorticity.
fig = panel_wise_visualization(ζ, colorrange=ζ_Array_color_range)
save("vorticity.png", fig)

# Plot the MITgcm zonal velocity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, Us, hide_decorations = false, colorrange = u_Array_color_range, colormap = :balance)
save("zonal_velocity_MITgcm.png", fig)

# Plot the MITgcm meridional velocity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, Vs, hide_decorations = false, colorrange = v_Array_color_range, colormap = :balance)
save("meridional_velocity_MITgcm.png", fig)

# Plot the MITgcm vorticity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, momVort3s, hide_decorations = false, colorrange = momVort3s_color_range, colormap = :balance)
save("vorticity_MITgcm.png", fig)

# Plot the difference between the Oceananigans zonal velocity and the MITgcm zonal velocity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, u_difference, hide_decorations = false, colorrange = u_difference_color_range, colormap = :balance)
save("zonal_velocity_difference.png", fig)

# Plot the difference between the Oceananigans meridional velocity and the MITgcm meridional velocity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, v_difference, hide_decorations = false, colorrange = v_difference_color_range, colormap = :balance)
save("meridional_velocity_difference.png", fig)

# Plot the difference between the Oceananigansvorticity and the MITgcm vorticity.
fig = panel_wise_visualization_MITgcm(XGs, YGs, momVort3s_difference, hide_decorations = false, colorrange = momVort3s_difference_color_range, colormap = :balance)
save("vorticity_difference.png", fig)