for ds in 1 2 3
do
  for i in {1..10}
  do
    for stb in 0 1
    do
      python train.py --scnd-seed $ds --main-seed $i --stb $stb --dry-run 0 --config engine.json
      # apptainer exec --nv neural_stability.sif python train.py --scnd-seed $ds --main-seed $i --stb $stb --dry-run 0 --config engine.json
    done
  done
done