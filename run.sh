# Decision Transformer (DT)
python -m atari.run_dt_atari --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --num_steps 100000 --num_buffers 50 --game 'Breakout' --batch_size 128
