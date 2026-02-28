CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 PYTHONWARNINGS="ignore" \
python libero/lifelong/main.py seed=10000 \
    benchmark_name=LIBERO_GOAL \
    policy=bc_transformer_policy \
    lifelong=base \