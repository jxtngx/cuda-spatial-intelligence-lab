---
name: sagemaker-bedrock-deploy
description: Deploy a model trained on DGX Spark to AWS — either as a SageMaker BYOC real-time/async endpoint, or as a Bedrock custom-model import (for supported architectures). Covers ECR, container contract, autoscaling, and verifying the same model produces consistent outputs as the local Spark Triton deployment.
---

# Deploying to AWS SageMaker / Bedrock

Companion to `sagemaker-bedrock-deploy`. Used in Week 14 (dual deploy).

## Decision: SageMaker vs Bedrock

| Factor | SageMaker BYOC | Bedrock custom import |
|---|---|---|
| Architecture support | Anything you can package | Restricted to supported base architectures (currently centered on Llama-family, Mistral, etc. — verify) |
| GPU choice | You pick the instance (`ml.g6*` Ada, `ml.p5*` Hopper, etc.) | Managed; you pay per token / per provisioned throughput |
| Engine portability | Build the TRT engine *inside* the container, targeting the instance GPU | Bedrock manages serving |
| When to use | Cosmos-Predict diffusion, custom CUDA backends, any non-LLM | LLM-class fine-tunes whose base is supported |

If you don't know whether Bedrock supports your base architecture today,
**check the AWS docs** — that list moves. Default to SageMaker BYOC.

## SageMaker BYOC workflow

### 1. Container contract

SageMaker calls two endpoints:

- `GET /ping` — return 200 if healthy.
- `POST /invocations` — request body is whatever your client sends;
  return the prediction in your chosen format (typically JSON).

Minimal Flask shim (or FastAPI):

```python
from fastapi import FastAPI, Request
import uvicorn
app = FastAPI()

@app.get("/ping")
def ping(): return {"status": "ok"}

@app.post("/invocations")
async def invoke(req: Request):
    body = await req.json()
    return {"prediction": run_model(body)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

`Dockerfile` essentials:

```dockerfile
FROM nvcr.io/nvidia/tensorrt:25.10-py3
WORKDIR /opt/program
COPY app.py model_loader.py engine.plan ./
RUN pip install fastapi uvicorn
ENV PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
```

### 2. Build TRT engine for the target GPU

`sm_121` (Spark Blackwell) engines **do not** run on AWS. Inside your
SageMaker container build, target the instance GPU:

- `ml.p5*` (H100) → `sm_90`.
- `ml.p4d*` (A100) → `sm_80`.
- `ml.g6*` (L40S / Ada) → `sm_89`.
- `ml.g6e*` (L4 Ada) → `sm_89`.

Build the engine in a multi-stage Dockerfile so you don't ship the ONNX
to production:

```dockerfile
FROM nvcr.io/nvidia/tensorrt:25.10-py3 AS build
COPY model.onnx /tmp/
RUN trtexec --onnx=/tmp/model.onnx \
    --saveEngine=/tmp/engine.plan \
    --fp16 --useCudaGraph \
    --builderOptimizationLevel=5

FROM nvcr.io/nvidia/tensorrt:25.10-py3
COPY --from=build /tmp/engine.plan /opt/program/
COPY app.py /opt/program/
WORKDIR /opt/program
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]
```

### 3. Push to ECR

```bash
ACCT=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1
REPO=cosmos-predict
aws ecr describe-repositories --repository-names $REPO --region $REGION \
  || aws ecr create-repository --repository-name $REPO --region $REGION
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCT.dkr.ecr.$REGION.amazonaws.com
docker buildx build --platform linux/amd64 -t $REPO:latest .
docker tag $REPO:latest $ACCT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
docker push $ACCT.dkr.ecr.$REGION.amazonaws.com/$REPO:latest
```

### 4. Create endpoint

```python
import sagemaker
from sagemaker.model import Model
sess = sagemaker.Session()
role = "arn:aws:iam::...:role/SageMakerExecutionRole"
m = Model(
    image_uri=f"{acct}.dkr.ecr.{region}.amazonaws.com/{repo}:latest",
    role=role,
    sagemaker_session=sess,
)
predictor = m.deploy(
    initial_instance_count=1,
    instance_type="ml.g6e.xlarge",
    endpoint_name="cosmos-predict-prod",
)
```

### 5. Smoke-test

```python
import boto3, json
runtime = boto3.client("sagemaker-runtime", region_name="us-east-1")
resp = runtime.invoke_endpoint(
    EndpointName="cosmos-predict-prod",
    ContentType="application/json",
    Body=json.dumps({"prompt": "..."}),
)
print(resp["Body"].read())
```

### 6. Async endpoint (alternative)

If your inference takes more than ~30 s (likely for diffusion), use a
SageMaker **async endpoint** with S3 input/output instead of real-time.
Same container; different deploy config.

## Bedrock custom model import (when applicable)

1. Upload weights (Hugging Face format) to S3.
2. Create an import job:
   ```bash
   aws bedrock create-model-import-job \
     --job-name cosmos-reason-v1 \
     --imported-model-name cosmos-reason-v1 \
     --role-arn arn:aws:iam::...:role/BedrockImportRole \
     --model-data-source s3DataSource={s3Uri=s3://my-bucket/weights/}
   ```
3. Wait for `Completed`. Attach a Provisioned Throughput unit.
4. Invoke via `bedrock-runtime`'s `Converse` or `InvokeModel`.

## Validation across both targets

End of Week 14: a small Python harness (`bench/dual_deploy_bench.py`) that:
1. Sends the same N requests to Spark Triton and SageMaker endpoint.
2. Reports p50/p95 latency, throughput, $-per-1k for AWS, kWh-est. for
   Spark.
3. Asserts output equivalence (cosine similarity for embeddings; BLEU/
   exact-match for text; SSIM/PSNR for images).
