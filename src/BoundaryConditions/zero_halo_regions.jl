function zero_halo_regions!(fields::Union{Tuple,NamedTuple})
    for field in fields
      zero_halo_regions!(field)
    end
    return nothing
end

zero_halo_regions!(field) = zero_halo_regions!(field.data, field.grid)

function zero_halo_regions!(c::AbstractArray, grid)
      zero_west_halo!(c, grid.Hx, grid.Nx)
      zero_east_halo!(c, grid.Hx, grid.Nx)
     zero_south_halo!(c, grid.Hy, grid.Ny)
     zero_north_halo!(c, grid.Hy, grid.Ny)
       zero_top_halo!(c, grid.Hz, grid.Nz)
    zero_bottom_halo!(c, grid.Hz, grid.Nz)
    return nothing
end

  zero_west_halo!(c, H, N) = @views @. c[1:H, :, :] = 0
 zero_south_halo!(c, H, N) = @views @. c[:, 1:H, :] = 0
zero_bottom_halo!(c, H, N) = @views @. c[:, :, 1:H] = 0

 zero_east_halo!(c, H, N) = @views @. c[N+H+1:N+2H, :, :] = 0
zero_north_halo!(c, H, N) = @views @. c[:, N+H+1:N+2H, :] = 0
  zero_top_halo!(c, H, N) = @views @. c[:, :, N+H+1:N+2H] = 0
