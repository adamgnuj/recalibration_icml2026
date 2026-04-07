using EnergyStatistics

function energy_distance(pw, y_in_interval)
    ed = mean(pw[y_in_interval, .!y_in_interval]) 
    ed -= mean(pw[y_in_interval, y_in_interval]) / 2 
    ed -= mean(pw[.!y_in_interval, .!y_in_interval]) / 2
    ed
end

_safe_sqrt(x::Real) = x >= zero(x) ? sqrt(x) : zero(x)

function test_independence(intervals::AbstractMatrix, y_in_interval, Z_PIT; n_bootstrap = 500)
    D_int = EnergyStatistics.dcenter!(EnergyStatistics.DistanceMatrix(Float64, eachrow(Array(intervals)), norm))
    D_pit = EnergyStatistics.dcenter!(EnergyStatistics.DistanceMatrix(Float64, Z_PIT, norm))
    dcor_value = dcor(D_int, D_pit)
    dcor_value
end

function eval_interval_result(intervals, obs_test, Z_PIT)
    # test coverage
    _y_upper_check = obs_test .<= intervals[:, 2]
    _y_lower_check = obs_test .>= intervals[:, 1]
    y_in_interval = _y_upper_check .& _y_lower_check
    coverage = mean(y_in_interval)

    # compute average interval length:
    average_int_len = mean(intervals[:, 2] - intervals[:, 1])
    median_int_len = median(intervals[:, 2] - intervals[:, 1])

    # test error cancellation
     dcor_value = test_independence(intervals, y_in_interval, Z_PIT)

    coverage, average_int_len, median_int_len, dcor_value
end