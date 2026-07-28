#!/bin/bash
#conda activate reward-machines-repoman
cd ../reward_machines

#TRAIN

for i in `seq 0 5`; 
do
	(

python run.py --env=Boxman-BlueSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueSquareTask-v0/$i/
python run.py --env=Boxman-BlueSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueSquareTask-v0/$i/

python run.py --env=Boxman-PurpleCircleTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-PurpleCircleTask-v0/$i/
python run.py --env=Boxman-PurpleCircleTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-PurpleCircleTask-v0/$i/

python run.py --env=Boxman-BlueNotSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueNotSquareTask-v0/$i/
python run.py --env=Boxman-BlueNotSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueNotSquareTask-v0/$i/

python run.py --env=Boxman-BlueSquareAndPurpleCircleTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueSquareAndPurpleCircleTask-v0/$i/
python run.py --env=Boxman-BlueSquareAndPurpleCircleTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueSquareAndPurpleCircleTask-v0/$i/

python run.py --env=Boxman-BlueOrPurpleAndSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/Boxman-BlueOrPurpleAndSquareTask-v0/0/
python run.py --env=Boxman-BlueOrPurpleAndSquareTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/Boxman-BlueOrPurpleAndSquareTask-v0/0/

python run.py --env=OfficeGetCoffeeTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/OfficeGetCoffeeTask-v0/$i/
python run.py --env=OfficeGetCoffeeTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/OfficeGetCoffeeTask-v0/$i/

python run.py --env=OfficeGoAandDTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/OfficeGoAandDTask-v0/$i/
python run.py --env=OfficeGoAandDTask-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/OfficeGoAandDTask-v0/$i/

python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --num_timesteps=1e5 --gamma=0.9 --alg=qlearning --use_crm --use_rs --save_path=data/ql-crm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/$i/
python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --num_timesteps=1e5 --gamma=0.9 --alg=hrm --use_rs --save_path=data/hrm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/$i/
	) &> log &
done
