# Decision Transformer (DT)
num_steps=100000

python -m atari.run_dt_atari --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --model_name up-causal --num_steps ${num_steps} --num_buffers 50 --game 'Breakout' --batch_size 128 2>&1 | tee up-causal.log

# python -m atari.run_dt_atari --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --model_name reformer --num_steps ${num_steps} --num_buffers 50 --game 'Breakout' --batch_size 128 2>&1 | tee reformer.log

python -m atari.run_dt_atari --seed 123 --context_length 30 --epochs 5 --model_type 'reward_conditioned' --model_name gpt --num_steps ${num_steps} --num_buffers 50 --game 'Breakout' --batch_size 128 2>&1 | tee gpt.log
