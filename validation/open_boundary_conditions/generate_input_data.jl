
#####
##### Input file setup: has to be done only once!
#####

function generate_input_data!(grid, times, filename)

    T_top_tmp = Field((Center, Center, Nothing), grid)   
    west_tmp  = Field((Nothing, Center, Center), grid)   
    
    # Fictitious data
    Tₜ(y, t) = 2 + 20 * cos(2π * (y - 15) / 60) * sin(π * (t - 365days) / 365days)

    T_top = FieldTimeSeries((Center, Center, Nothing), grid, times; 
                                    backend = OnDisk(),
                                    path = filename,
                                    name = "T_top")

    T_west = FieldTimeSeries((Nothing, Center, Center), grid, times; 
                                    backend = OnDisk(),
                                    path = filename,
                                    name = "T_west")

    u_west = FieldTimeSeries((Nothing, Center, Center), grid, times; 
                                    backend = OnDisk(),
                                    path = filename,
                                    name = "u_west")

    # write down the data on file
    for t in eachindex(times)
        @info "writing down data for time $t"
        set!(T_top_tmp, (x, y) -> Tₜ(y, times[t]))
        set!(T_top, T_top_tmp, t)

        set!(west_tmp, (y, z) -> sin((y - 15) / 60 * 2π) * cos(times[t] / times[end] * 2π - π/2))
        set!(u_west, west_tmp, t)
        
        set!(west_tmp, (y, z) -> Tₜ(y, times[t]) * (1 + z / 1000))
        set!(T_west, west_tmp, t)
    end
end