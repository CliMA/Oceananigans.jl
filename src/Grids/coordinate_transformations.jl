using CubedSphere.SphericalGeometry: lat_lon_to_x, lat_lon_to_y, lat_lon_to_z
using GPUArraysCore: @allowscalar

longitude_in_same_window(λ₁, λ₂) = mod(λ₁ - λ₂ + 180, 360) + λ₂ - 180

flip_location(::Center) = Face()
flip_location(::Face) = Center()

"""
    get_longitude_vertices(i, j, k, grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

Return the longitudes that correspond to the four vertices of cell `i, j, k` at location `(ℓx, ℓy, ℓz)`. The first
vertex is the cell's Southern-Western one and the rest follow in counter-clockwise order.
"""
function get_longitude_vertices(i, j, k, grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

    if ℓx == Center()
        i₀ = i
    elseif ℓx == Face()
        i₀ = i-1
    end

    if ℓy == Center()
        j₀ = j
    elseif ℓy == Face()
        j₀ = j-1
    end

    λ₁ = λnode( i₀,   j₀,  k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    λ₂ = λnode(i₀+1,  j₀,  k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    λ₃ = λnode(i₀+1, j₀+1, k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    λ₄ = λnode( i₀,  j₀+1, k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)

    return [λ₁; λ₂; λ₃; λ₄]
end

"""
    get_latitude_vertices(i, j, k, grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

Return the latitudes that correspond to the four vertices of cell `i, j, k` at location `(ℓx, ℓy, ℓz)`. The first vertex
is the cell's Southern-Western one and the rest follow in counter-clockwise order.
"""
function get_latitude_vertices(i, j, k, grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

    if ℓx == Center()
        i₀ = i
    elseif ℓx == Face()
        i₀ = i-1
    end

    if ℓy == Center()
        j₀ = j
    elseif ℓy == Face()
        j₀ = j-1
    end

    φ₁ = φnode( i₀,   j₀,  k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    φ₂ = φnode(i₀+1,  j₀,  k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    φ₃ = φnode(i₀+1, j₀+1, k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)
    φ₄ = φnode( i₀,  j₀+1, k, grid, flip_location(ℓx), flip_location(ℓy), ℓz)

    return [φ₁; φ₂; φ₃; φ₄]
end

"""
    get_lat_lon_nodes_and_vertices(grid, ℓx, ℓy, ℓz)

Return the latitude-longitude coordinates of the horizontal nodes of the `grid` at locations `ℓx`, `ℓy`, and `ℓz` and
also the coordinates of the four vertices that determine the cell surrounding each node.

See [`get_longitude_vertices`](@ref) and [`get_latitude_vertices`](@ref).
"""
function get_lat_lon_nodes_and_vertices(grid, ℓx, ℓy, ℓz)

    TX, TY, _ = topology(grid)

    λ = zeros(eltype(grid), total_length(ℓx, TX(), grid.Nx, 0), total_length(ℓy, TY(), grid.Ny, 0))
    φ = zeros(eltype(grid), total_length(ℓx, TX(), grid.Nx, 0), total_length(ℓy, TY(), grid.Ny, 0))

    for j in axes(λ, 2), i in axes(λ, 1)
        @allowscalar λ[i, j] = λnode(i, j, 1, grid, ℓx, ℓy, ℓz)
        @allowscalar φ[i, j] = φnode(i, j, 1, grid, ℓx, ℓy, ℓz)
    end

    λvertices = zeros(4, size(λ)...)
    φvertices = zeros(4, size(φ)...)

    for j in axes(λ, 2), i in axes(λ, 1)
        @allowscalar λvertices[:, i, j] = get_longitude_vertices(i, j, 1, grid, ℓx, ℓy, ℓz)
        @allowscalar φvertices[:, i, j] =  get_latitude_vertices(i, j, 1, grid, ℓx, ℓy, ℓz)
    end

    λ = mod.(λ .+ 180, 360) .- 180
    λvertices = longitude_in_same_window.(λvertices, reshape(λ, (1, size(λ)...)))

    return (φ, λ), (φvertices, λvertices)
end

"""
    get_cartesian_nodes_and_vertices(grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

Return the cartesian coordinates of the horizontal nodes of the `grid` at locations `ℓx`, `ℓy`, and `ℓz` on the unit
sphere and also the corresponding coordinates of the four vertices that determine the cell surrounding each node.

See [`get_lat_lon_nodes_and_vertices`](@ref).
"""
function get_cartesian_nodes_and_vertices(grid::Union{LatitudeLongitudeGrid, OrthogonalSphericalShellGrid}, ℓx, ℓy, ℓz)

    (φ, λ), (φvertices, λvertices) = get_lat_lon_nodes_and_vertices(grid, ℓx, ℓy, ℓz)

    x = similar(λ)
    y = similar(λ)
    z = similar(λ)

    xvertices = similar(λvertices)
    yvertices = similar(λvertices)
    zvertices = similar(λvertices)

    for j in axes(λ, 2), i in axes(λ, 1)
        x[i, j] = lat_lon_to_x(φ[i, j], λ[i, j])
        y[i, j] = lat_lon_to_y(φ[i, j], λ[i, j])
        z[i, j] = lat_lon_to_z(φ[i, j])

        for vertex in 1:4
            xvertices[vertex, i, j] = lat_lon_to_x(φvertices[vertex, i, j], λvertices[vertex, i, j])
            yvertices[vertex, i, j] = lat_lon_to_y(φvertices[vertex, i, j], λvertices[vertex, i, j])
            zvertices[vertex, i, j] = lat_lon_to_z(φvertices[vertex, i, j])
        end
    end

    return (x, y, z), (xvertices, yvertices, zvertices)
end
