python3 "forecast/run_forecast.py" --load_tokenized_data false --n_epochs 10 --num_lags 0

python3 "forecast/run_forecast.py" --load_tokenized_data false --n_epochs 10 --num_lags 1

python3 "forecast/run_forecast.py" --load_tokenized_data false --n_epochs 10 --num_lags 2

python3 "forecast/run_forecast.py" --load_tokenized_data false --n_epochs 10 --num_lags 3

python3 "forecast/run_forecast.py" --load_tokenized_data false --n_epochs 10 --num_lags 4


python3 "forecast/run_forecast.py" --model roberta-large --bs 4 --load_tokenized_data false --n_epochs 10 --num_lags 0

python3 "forecast/run_forecast.py" --model roberta-large --bs 4 --load_tokenized_data false --n_epochs 10 --num_lags 1

python3 "forecast/run_forecast.py" --model roberta-large --bs 4 --load_tokenized_data false --n_epochs 10 --num_lags 2

python3 "forecast/run_forecast.py" --model roberta-large --bs 4 --load_tokenized_data false --n_epochs 10 --num_lags 3

python3 "forecast/run_forecast.py" --model roberta-large --bs 4 --load_tokenized_data false --n_epochs 10 --num_lags 4
