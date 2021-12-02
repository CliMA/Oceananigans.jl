using Oceananigans.Utils: TimeInterval, IterationInterval, WallTimeInterval, SpecifiedTimes
using Oceananigans.TimeSteppers: Clock

@testset "Schedules" begin
    @info "Testing schedules..."

    # Some fake models
    fake_model_at_iter_3 = (; clock=Clock(time=0.0, iteration=3))
    fake_model_at_iter_5 = (; clock=Clock(time=0.0, iteration=5))

    fake_model_at_time_2 = (; clock=Clock(time=2.0, iteration=0))
    fake_model_at_time_3 = (; clock=Clock(time=3.0, iteration=0))
    fake_model_at_time_4 = (; clock=Clock(time=4.0, iteration=0))
    fake_model_at_time_5 = (; clock=Clock(time=5.0, iteration=0))

    # TimeInterval
    ti = TimeInterval(2)
    @test ti.interval == 2.0
    @test ti(fake_model_at_time_2)
    @test !(ti(fake_model_at_time_3))

    # IterationInterval
    ii = IterationInterval(3)

    @test !(ii(fake_model_at_iter_5))
    @test ii(fake_model_at_iter_3)

    # WallTimeInterval
    wti = WallTimeInterval(1e-9)

    @test wti.interval == 1e-9
    @test wti(nothing)

    # SpecifiedTimes
    st = st_list = SpecifiedTimes(2, 5, 6)
    st_vector = SpecifiedTimes([2, 5, 6])
    @test st_list.times == st_vector.times
    @test st.times == [2.0, 5.0, 6.0]

    # Times are sorted
    st = SpecifiedTimes(5, 2, 6)
    @test st.times == [2.0, 5.0, 6.0]

    @test st(fake_model_at_time_2)

    @test !(st(fake_model_at_time_4))
    @test st(fake_model_at_time_5)
end
