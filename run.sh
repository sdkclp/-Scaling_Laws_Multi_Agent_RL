episode_collect=10
repeat_collect=2
step_epoch=1000
epoch=1000
hidden="64 64 64"

for i in 1 2 3 4 5 
do
  nohup python tag.py --seed ${i}  --episode-per-collect $episode_collect --repeat-per-collect $repeat_collect --step-per-epoch $step_epoch --epoch $epoch --hidden-sizes $hidden &
done

