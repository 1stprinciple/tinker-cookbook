Use existing training shapes

Find one from
firectl list training-shapes


Create trainer instance and deployment instance
python tinker_cookbook/fireworks/setup.py  
which will pick up the config from fireworks.yaml


Run any RL job:

Just attach these 3 lines, replace these with your own instances.
base_url="https://api.fireworks.ai/training/v1/rlorTrainerJobs/pyroworks/dpiozbv7yjv9z827" \
    fireworks_deployment_id=qwen3-4b-instruct-2507-1774951799  \
    fireworks_base_model_name=accounts/fireworks/models/qwen3-4b-instruct-2507