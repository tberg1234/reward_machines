#!/bin/bash
#conda activate reward-machines-repoman
cd ../reward_machines

#RUN
python run.py --env=Boxman-BlueSquareTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueSquareTask-v0/0/ --run_model
python run.py --env=Boxman-BlueSquareTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueSquareTask-v0/0/ --run_model

python run.py --env=Boxman-PurpleCircleTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-PurpleCircleTask-v0/0/ --run_model
python run.py --env=Boxman-PurpleCircleTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-PurpleCircleTask-v0/0/ --run_model

python run.py --env=Boxman-BlueNotSquareTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueNotSquareTask-v0/0/ --run_model
python run.py --env=Boxman-BlueNotSquareTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueNotSquareTask-v0/0/ --run_model

python run.py --env=Boxman-BlueSquareAndPurpleCircleTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueSquareAndPurpleCircleTask-v0/0/ --run_model
python run.py --env=Boxman-BlueSquareAndPurpleCircleTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueSquareAndPurpleCircleTask-v0/0/ --run_model

python run.py --env=Boxman-BlueOrPurpleAndSquareTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueOrPurpleAndSquareTask-v0/0/ --run_model
python run.py --env=Boxman-BlueOrPurpleAndSquareTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueOrPurpleAndSquareTask-v0/0/ --run_model

python run.py --env=OfficeGetCoffeeTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeGetCoffeeTask-v0/0/ --run_model
python run.py --env=OfficeGetCoffeeTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeGetCoffeeTask-v0/0/ --run_model

python run.py --env=OfficeGoAandDTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeGoAandDTask-v0/0/ --run_model
python run.py --env=OfficeGoAandDTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeGoAandDTask-v0/0/ --run_model

python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/0/ --run_model
python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/0/ --run_model