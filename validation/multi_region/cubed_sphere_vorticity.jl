using Oceananigans, Printf

using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: replace_horizontal_vector_halos!
using Oceananigans.Grids: φnode, λnode, halo_size, total_size
using Oceananigans.MultiRegion: getregion, number_of_regions
using Oceananigans.Operators
using Oceananigans.Utils: Iterate

Nx = 9
Ny = 9
Nz = 1

Lz = 1
R = 1 # sphere's radius
U = 1 # velocity scale

grid = ConformalCubedSphereGrid(; panel_size = (Nx, Ny, Nz),
                                  z = (-Lz, 0),
                                  radius = R,
                                  horizontal_direction_halo = 2,
                                  partition = CubedSpherePartition(; R = 1))

Hx, Hy, Hz = halo_size(grid)

# Solid body rotation
φʳ = 90       # Latitude pierced by the axis of rotation
α  = 90 - φʳ  # Angle between axis of rotation and north pole (degrees)
ψᵣ(λ, φ, z) = - U * R * (sind(φ) * cosd(α) - cosd(λ) * cosd(φ) * sind(α))

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

for region in 1:number_of_regions(grid)
    u[region] .= - ∂y(ψ[region])
    v[region] .= + ∂x(ψ[region])
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
            # this is not correct -- we should not overwrite the above with these
            # u[region][Nx+1:Nx+Hx, Ny+1, k] .= u[region_north][1:Hx, 1, k]
            # v[region][Nx+1:Nx+Hx, Ny+1, k] .= v[region_north][1:Hy, 1, k]
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
    f[region] .= 2 * (U/R) * sind.(grid[region].φᶠᶠᵃ)
end

function error_ζ_ζtheoretical(ζ, ζtheoretical)
    grid = ζ.grid

    abs_error = Field{Face, Face, Center}(grid)
    for region in 1:number_of_regions(grid)
        parent(abs_error[region]) .= abs.(parent(ζtheoretical[region]) .- parent(ζ[region]))
    end

    rel_error = Field{Face, Face, Center}(grid)
    for region in 1:number_of_regions(grid)
        parent(rel_error[region]) .= parent(abs_error[region]) ./ abs.(parent(ζtheoretical[region]) .+ 100*eps(eltype(grid)))
    end

    return abs_error, rel_error
end

# using Imaginocean

using GLMakie

#=
# Imaginocean still doesn't work with Face-Face fields
fig = Figure(resolution = (2000, 2000), fontsize=30)

ax1 = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])
ax3 = Axis(fig[2, 1])
ax4 = Axis(fig[2, 2])

for region in 1:number_of_regions(grid)
    heatmap!(ax1, u, colorrange=(-1, 1), colormap = :balance)
    heatmap!(ax2, v, colorrange=(-1, 1), colormap = :balance)
    heatmap!(ax3, ψ, colorrange=(-1, 1), colormap = :balance)
    heatmap!(ax4, ζ, colorrange=(-1, 1), colormap = :balance)
end

fig
=#

function panel_wise_visualization(field, k=1; hide_decorations = true, colorrange = (-1, 1), colormap = :balance)

    fig = Figure(resolution = (1800, 1200))

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

fig = panel_wise_visualization(ψ)
save("streamfunction.png", fig)

fig = panel_wise_visualization(f, colorrange=(-2, 2))
save("f.png", fig)

fig = panel_wise_visualization(ζ, colorrange=(-2, 2))
save("vorticity.png", fig)

abs_error1, rel_error1 = error_ζ_ζtheoretical(ζ, f)

ζ2 = deepcopy(ζ)
recompute_vorticity_corners_using_interior_points!(ζ2)

fig = panel_wise_visualization(ζ2, colorrange=(-2, 2))
save("vorticity2.png", fig)

abs_error2, rel_error2 = error_ζ_ζtheoretical(ζ2, f)

fig = panel_wise_visualization(rel_error1, colorrange=(0, 0.8), colormap=:thermal)
save("rel_error1.png", fig)
fig = panel_wise_visualization(rel_error2, colorrange=(0, 0.8), colormap=:thermal)
save("rel_error2.png", fig)



using Oceananigans.Operators
using Oceananigans.Operators: Γᶠᶠᶜ, ζ₃ᶠᶠᶜ
using Oceananigans.Grids: lat_lon_to_cartesian, spherical_area_quadrilateral

relative_error_from_f(i, j,  region, ζ, f) = abs(ζ[region][i, j, 1] - f[region][i, j, 1]) / abs(f[region][i, j, 1])

# SouthEast
@show ζ[1][Nx+1, 1, 1]
@show f[1][Nx+1, 1, 1]
@show relative_error_from_f(1, 1, 1, ζ2, f)

Γse = - grid[1].Δyᶜᶠᵃ[Nx, 1] * v[1][Nx, 1, 1] - grid[2].Δxᶠᶜᵃ[1, 1] * u[2][1, 1, 1] + grid[2].Δyᶜᶠᵃ[1, 1] * v[2][1, 1, 1]

@show Γse
@show Γse - Γᶠᶠᶜ(Nx+1, 1, 1, grid[1], u[1], v[1])

a = lat_lon_to_cartesian(grid[1].φᶠᶠᵃ[Nx+1, 1], grid[1].λᶠᶠᵃ[Nx+1, 1], 1)
b = lat_lon_to_cartesian(grid[1].φᶠᶜᵃ[Nx+1, 1], grid[1].λᶠᶜᵃ[Nx+1, 1], 1)
c = lat_lon_to_cartesian(grid[1].φᶜᶜᵃ[Nx, 1], grid[1].λᶜᶜᵃ[Nx, 1], 1)
d = lat_lon_to_cartesian(grid[1].φᶜᶠᵃ[Nx, 1], grid[1].λᶜᶠᵃ[Nx  , 1], 1)
area_se = 3 * spherical_area_quadrilateral(a, b, c, d) * grid.radius^2

@show area_se - 3/4 * grid[1].Azᶠᶠᵃ[Nx+1, 1]

@show ζse = Γse / area_se
@show ζse - ζ₃ᶠᶠᶜ(Nx+1, 1, 1, grid[1], u[1], v[1])


@show ζ[1][1, 2, 1]
@show f[1][1, 2, 1]
@show relative_error_from_f(1, 2, 1, ζ2, f)

Γ12 = grid[1].Δyᶜᶠᵃ[1, 2] * v[1][1, 2, 1] - grid[1].Δyᶜᶠᵃ[0, 2] * v[1][0, 2, 1] +
      grid[1].Δxᶠᶜᵃ[1, 1] * u[1][1, 1, 1] - grid[1].Δxᶠᶜᵃ[1, 2] * u[1][1, 2, 1]

@show Γ12
@show Γ12 - Γᶠᶠᶜ(1, 2, 1, grid[1], u[1], v[1])

i = 1
j = 2
a = lat_lon_to_cartesian(grid[1].φᶠᶜᵃ[ i , j-1], grid[1].λᶠᶜᵃ[ i , j-1], 1)
b = lat_lon_to_cartesian(grid[1].φᶜᶜᵃ[ i , j-1], grid[1].λᶜᶜᵃ[ i , j-1], 1)
c = lat_lon_to_cartesian(grid[1].φᶜᶜᵃ[ i ,  j ], grid[1].λᶜᶜᵃ[ i ,  j ], 1)
d = lat_lon_to_cartesian(grid[1].φᶠᶜᵃ[ i ,  j ], grid[1].λᶠᶜᵃ[ i ,  j ], 1)

area_12 = 2 * spherical_area_quadrilateral(a, b, c, d) * grid.radius^2
@show area_12 - grid[1].Azᶠᶠᵃ[i, j]

ζ12 = Γ12 / grid[1].Azᶠᶠᵃ[1, 2]

@show ζ12
@show ζ12 - ζ₃ᶠᶠᶜ(1, 2, 1, grid[1], u[1], v[1])
