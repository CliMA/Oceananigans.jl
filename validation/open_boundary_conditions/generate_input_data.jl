
#####
##### Input file setup: has to be done only once!
#####

function generate_input_data!(grid, times, filename)

    T_tmp = Field((Center, Center, Nothing), grid)   
    u_tmp = Field((Face,   Center, Nothing), grid)

    # Fictitious data
    Tₜ(y, t) = 2 + 20 * cos(2π * y / 50) * sin(π * (t - 365days) / 365days)
    uₜ(y, t) = cos(π/2 * y / 50) * t / 365days

    T_top = FieldTimeSeries((Center, Center, Nothing), grid, times; 
                                    backend = OnDisk(),
                                    path = filename,
                                    name = "T_top")

    u_top = FieldTimeSeries((Face, Center, Nothing), grid, times; 
                                    backend = OnDisk(),
                                    path = filename,
                                    name = "u_top")

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
        set!(T_tmp, (x, y) -> Tₜ(y, times[t]))
        set!(u_tmp, (x, y) -> uₜ(y, times[t]))

        set!(T_top, T_tmp, t)
        set!(u_top, u_tmp, t) 
    end
end