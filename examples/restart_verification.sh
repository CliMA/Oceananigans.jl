#!/bin/bash
for script in *.jl; do
    casename=${script%.jl}
    echo $casename $script

    mkdir -p $casename
    cd $casename

    norestart="${casename}_0.jl"
    if [ ! -f "$norestart" ]; then
        cp ../$script $norestart
        sed -i '' '/^simulation =/s/stop_[^)]*)/stop_iteration=200)/' $norestart
        sed -i '' '/^run!/q' $norestart
        sed -i '' '/^run!/s/)$/, checkpoint_at_end=true)/' $norestart
        sed -i '' "/^run!/i\\"$'\n'"simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(100), prefix=\"norestart\")\\"$'\n'"" $norestart
        sed -i '' 's/^simulation.stop_iteration =/#simulation.stop_iteration =/' $restarted # to be safe, don't overwrite stop_iteration
    fi

    restarted="${casename}_1.jl"
    if [ ! -f "$restarted" ]; then
        cp ../$script $restarted
        sed -i '' '/^simulation =/s/stop_[^)]*)/stop_iteration=200)/' $restarted
        sed -i '' '/^run!/q' $restarted
        sed -i '' '/^run!/s/)$/, checkpoint_at_end=true)/' $restarted
        sed -i '' "/^run!/i\\"$'\n'"simulation.output_writers[:checkpointer] = Checkpointer(model, schedule=IterationInterval(100), prefix=\"restarted\")\\"$'\n'"" $restarted
        sed -i '' 's/^simulation.stop_iteration =/#simulation.stop_iteration =/' $restarted # to be safe, don't overwrite stop_iteration

        # restart specific
        sed -i '' 's/^set!/#set!/' $restarted # dont re-initialize
        sed -i '' '/^run!/s/)$/; pickup="norestart_iteration100.jld2")/' $restarted
    fi

    if [ ! -f 'norestart_iteration200.jld2' ]; then julia $norestart; fi
    if [ ! -f 'restarted_iteration200.jld2' ]; then julia $restarted; fi

    cd ..

    julia util/compare_checkpoints.jl $casename/norestart_iteration200.jld2 $casename/restarted_iteration200.jld2 2>&1 | tee compare_restart_$casename.log
done
