using Oceananigans.Grids: xnode, ynode

"""
    lat_lon_to_cartesian(longitude, latitude)

Convert `(longitude, latitude)` coordinates (in degrees) to
cartesian coordinates `(x, y, z)` on the unit sphere.
"""
lat_lon_to_cartesian(longitude, latitude) = (lat_lon_to_x(longitude, latitude),
                                             lat_lon_to_y(longitude, latitude),
                                             lat_lon_to_z(longitude, latitude))

"""
    lat_lon_to_x(longitude, latitude)

Convert `(longitude, latitude)` coordinates (in degrees) to
cartesian `x` on the unit sphere.
"""
lat_lon_to_x(longitude, latitude) = cosd(longitude) * cosd(latitude)

"""
    lat_lon_to_x(longitude, latitude)

Convert `(longitude, latitude)` coordinates (in degrees) to
cartesian `x` on the unit sphere.
"""
lat_lon_to_y(longitude, latitude) = sind(longitude) * cosd(latitude)

"""
    lat_lon_to_x(longitude, latitude)

Convert `(longitude, latitude)` coordinates (in degrees) to
cartesian `x` on the unit sphere.
"""
lat_lon_to_z(longitude, latitude) = sind(latitude)

longitude_in_same_window(λ1, λ2) = mod(λ1 - λ2 + 180, 360) + λ2 - 180

flip_location(::Center) = Face()
flip_location(::Face) = Center()

"""
    get_longitude_vertices(i, j, LX, LY, grid::OrthogonalSphericalShellGrid)

Return the longitudes that correspond to the four vertices of cell `i, j` at
position `(LX, LY)`. The first vertice is the cell's Southern-Western one
and the rest follow in counter-clockwise order.
"""
function get_longitude_vertices(i, j, LX, LY, grid::OrthogonalSphericalShellGrid)

    if LX == Center()
        i₀ = i
    elseif LX == Face()
        i₀ = i-1
    end

    if LY == Center()
        j₀ = j
    elseif LY == Face()
        j₀ = j-1
    end

    λ₁ = xnode(flip_location(LX), flip_location(LY), Center(),  i₀,   j₀,  1, grid)
    λ₂ = xnode(flip_location(LX), flip_location(LY), Center(), i₀+1,  j₀,  1, grid)
    λ₃ = xnode(flip_location(LX), flip_location(LY), Center(), i₀+1, j₀+1, 1, grid)
    λ₄ = xnode(flip_location(LX), flip_location(LY), Center(),  i₀,  j₀+1, 1, grid)

    return [λ₁; λ₂; λ₃; λ₄]
end

"""
    get_latitude_vertices(i, j, LX, LY, grid::OrthogonalSphericalShellGrid)

Return the latitudes that correspond to the four vertices of cell `i, j` at
position `(LX, LY)`. The first vertice is the cell's Southern-Western one
and the rest follow in counter-clockwise order.
"""
function get_latitude_vertices(i, j, LX, LY, grid::OrthogonalSphericalShellGrid)

    if LX == Center()
        i₀ = i
    elseif LX == Face()
        i₀ = i-1
    end

    if LY == Center()
        j₀ = j
    elseif LY == Face()
        j₀ = j-1
    end

    φ₁ = ynode(flip_location(LX), flip_location(LY), Center(),  i₀,   j₀,  1, grid)
    φ₂ = ynode(flip_location(LX), flip_location(LY), Center(), i₀+1,  j₀,  1, grid)
    φ₃ = ynode(flip_location(LX), flip_location(LY), Center(), i₀+1, j₀+1, 1, grid)
    φ₄ = ynode(flip_location(LX), flip_location(LY), Center(),  i₀,  j₀+1, 1, grid)

    return [φ₁; φ₂; φ₃; φ₄]
end

"""
    get_lat_lon_nodes_and_vertices(LX, LY, grid::OrthogonalSphericalShellGrid)

Return the latitude-longitude coordinates of the horizontal nodes of the
`grid` at locations `LX` and `LY` and also the coordinates of the four vertices
that determine the cell surrounding each node.

See [`get_longitude_vertices`](@ref) and [`get_latitude_vertices`](@ref).
"""
function get_lat_lon_nodes_and_vertices(LX, LY, grid::OrthogonalSphericalShellGrid)
    λ = xnodes((typeof(LX), typeof(LY)), grid)
    φ = ynodes((typeof(LX), typeof(LY)), grid)

    nλ, nφ = size(λ)

    λvertices = zeros(4, nλ, nφ)
    φvertices = zeros(4, nλ, nφ)

    for j in 1:nφ, i in 1:nλ
        λvertices[:, i, j] = get_longitude_vertices(i, j, LX, LY, grid)
        φvertices[:, i, j] =  get_latitude_vertices(i, j, LX, LY, grid)
    end

    λ = mod.(λ .+ 180, 360) .- 180
    λvertices = longitude_in_same_window.(λvertices, reshape(λ, (1, size(λ)...)))
    
    return (λ, φ), (λvertices, φvertices)
end

"""
    get_cartesian_nodes_and_vertices(LX, LY, grid::OrthogonalSphericalShellGrid)

Return the cartesian coordinates of the horizontal nodes of the `grid`
at locations `LX` and `LY` on the unit sphere and also the corresponding
coordinates of the four vertices that determine the cell surrounding each
node.

See [`get_lat_lon_nodes_and_vertices`](@ref).
"""
function get_cartesian_nodes_and_vertices(LX, LY, grid::OrthogonalSphericalShellGrid)

    (λ, φ), (λvertices, φvertices) = get_lat_lon_nodes_and_vertices(LX, LY, grid)

    x = similar(λ)
    y = similar(λ)
    z = similar(λ)

    xvertices = similar(λvertices)
    yvertices = similar(λvertices)
    zvertices = similar(λvertices)

    nλ, nφ = size(λ)

    for j in 1:nφ, i in 1:nλ
        x[i, j] = lat_lon_to_x(λ[i, j], φ[i, j])
        y[i, j] = lat_lon_to_y(λ[i, j], φ[i, j])
        z[i, j] = lat_lon_to_z(λ[i, j], φ[i, j])
        
        for vertex in 1:4
            xvertices[vertex, i, j] = lat_lon_to_x(λvertices[vertex, i, j], φvertices[vertex, i, j])
            yvertices[vertex, i, j] = lat_lon_to_y(λvertices[vertex, i, j], φvertices[vertex, i, j])
            zvertices[vertex, i, j] = lat_lon_to_z(λvertices[vertex, i, j], φvertices[vertex, i, j])
        end
    end
    
    return (x, y, z), (xvertices, yvertices, zvertices)
end
