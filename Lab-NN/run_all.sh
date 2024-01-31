# Code for part II / III /IV
# Make Sure you want to handin , then run this script!

set +x

# python MLPAE_train_script.py > ./log/train_MLPAE.txt

# python visualization.py --model MLPAE


# python AE_train_script.py --model AE > ./log/train_AE.txt

# python visualization.py --model AE

# python random_generation.py --model AE

python VAE_train_script.py --model VAE > ./log/train_VAE.txt

python visualization.py --model VAE

python random_generation.py --model VAE

