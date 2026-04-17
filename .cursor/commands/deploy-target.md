---
description: Deploy or refresh the current week's model to a target — spark | sagemaker | bedrock. Delegates to model-deployer with the appropriate workflow from the sagemaker-bedrock-deploy skill.
---

The user invoked `/deploy-target` with an argument:
`spark` | `sagemaker` | `bedrock`. If missing, ask once.

Workflow:

1. Identify the model artifact in scope (current week's `report/` or a
   path the user names).
2. Delegate to **`model-deployer`** with:
   > "Deploy the model artifact at `<path>` to target `<target>`. Use
   > the workflow from `.cursor/skills/sagemaker-bedrock-deploy/SKILL.md`
   > (and Triton workflow from the `model-deployer` agent system prompt
   > for the `spark` target). Produce: build commands, the resulting
   > image / engine path, the deploy command, and a smoke-test snippet.
   > Reject the request if the target doesn't make sense for the model
   > architecture (e.g. Bedrock for Cosmos-Predict diffusion — note
   > this and recommend SageMaker BYOC instead)."
3. Print the deployer's output verbatim.
4. After a successful deploy, recommend running the dual-deploy bench
   (`bench/dual_deploy_bench.py`) to compare cost/latency vs the other
   target.
