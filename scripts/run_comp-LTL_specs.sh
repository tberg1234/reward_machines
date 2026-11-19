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

# Spec 1: F(b&e) & G!(c|a|d) - Eventually blue AND square, avoid purple/white/circle
python run.py --env=Boxman-BlueSquareSafe-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueSquareSafe-v0/0/ --run_model
python run.py --env=Boxman-BlueSquareSafe-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueSquareSafe-v0/0/ --run_model

# Spec 2: F(c&d) & G!(b|a|e) - Eventually purple AND circle, avoid blue/white/square
python run.py --env=Boxman-PurpleCircleSafe-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-PurpleCircleSafe-v0/0/ --run_model
python run.py --env=Boxman-PurpleCircleSafe-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-PurpleCircleSafe-v0/0/ --run_model

# Spec 3: F(b) & G!(c|a|e) - Eventually blue, avoid purple/white/square
python run.py --env=Boxman-BlueSafe-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-BlueSafe-v0/0/ --run_model
python run.py --env=Boxman-BlueSafe-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-BlueSafe-v0/0/ --run_model

# Spec 4: G(F(b&e)) & G(F(c&d)) & G!a - Patrol both, avoid white
python run.py --env=Boxman-PatrolBothAvoidWhite-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-PatrolBothAvoidWhite-v0/0/ --run_model
python run.py --env=Boxman-PatrolBothAvoidWhite-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-PatrolBothAvoidWhite-v0/0/ --run_model

# Spec 5: F(G((b|c)&e)) & G!(a|d) - Eventually always (blue OR purple) AND square, avoid white/circle
python run.py --env=Boxman-StayBlueOrPurpleSquare-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/Boxman-StayBlueOrPurpleSquare-v0/0/ --run_model
python run.py --env=Boxman-StayBlueOrPurpleSquare-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/Boxman-StayBlueOrPurpleSquare-v0/0/ --run_model

python run.py --env=OfficeGetCoffeeTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeGetCoffeeTask-v0/0/ --run_model
python run.py --env=OfficeGetCoffeeTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeGetCoffeeTask-v0/0/ --run_model

python run.py --env=OfficeGoAandDTask-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeGoAandDTask-v0/0/ --run_model
python run.py --env=OfficeGoAandDTask-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeGoAandDTask-v0/0/ --run_model

python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --alg=qlearning --use_crm --use_rs --load_path=../../data/ql-crm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/0/ --run_model
python run.py --env=OfficeAvoidBAndIn2TasksGetDecorator-v0 --alg=hrm --use_rs --load_path=../../data/hrm-rs/OfficeAvoidBAndIn2TasksGetDecorator-v0/0/ --run_model