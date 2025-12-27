# Kubernetes Deployment for Spam Detector

This directory contains Kubernetes manifests for deploying the spam detector API on minikube.

## Prerequisites

- Minikube installed and running
- kubectl configured
- Docker installed
- Trained model files available

## Deployment Steps

### 1. Start Minikube

```bash
minikube start
```

### 2. Build Docker Image

Build the Docker image and load it into minikube:

```bash
# Use minikube's Docker daemon
eval $(minikube docker-env)

# Build the image
docker build -t spam-detector:latest .
```

### 3. Prepare Model Files

First, ensure you have trained models. Copy them to the PVC:

```bash
# Create a temporary pod to upload models
kubectl run model-uploader --image=busybox --restart=Never --command -- sleep 3600

# Copy model files
kubectl cp models/spam_detector_model.pkl model-uploader:/tmp/
kubectl cp models/vectorizer.pkl model-uploader:/tmp/

# Clean up
kubectl delete pod model-uploader
```

### 4. Deploy to Kubernetes

Apply all manifests:

```bash
# Create PVC for models
kubectl apply -f k8s/pvc.yaml

# Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# Create Deployment
kubectl apply -f k8s/deployment.yaml

# Create Service
kubectl apply -f k8s/service.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=spam-detector

# Check service
kubectl get svc spam-detector

# Check logs
kubectl logs -l app=spam-detector
```

### 6. Access the API

Get the service URL:

```bash
minikube service spam-detector --url
```

Or use port forwarding:

```bash
kubectl port-forward svc/spam-detector 8000:8000
```

Then access the API at `http://localhost:8000`

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Predict single email
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "CONGRATULATIONS! You won a prize!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Meeting tomorrow at 2pm", "Click here to claim your prize!"]}'
```

## Scaling

Scale the deployment:

```bash
kubectl scale deployment spam-detector --replicas=3
```

## Monitoring

```bash
# Watch pods
kubectl get pods -l app=spam-detector -w

# Stream logs
kubectl logs -f -l app=spam-detector

# Get resource usage
kubectl top pods -l app=spam-detector
```

## Cleanup

```bash
kubectl delete -f k8s/
```

## Notes

- The deployment uses a PersistentVolumeClaim to store model files
- Models are mounted as read-only volumes
- Health checks ensure the API is responding
- Resource limits prevent pods from consuming too many resources
- The service is exposed via NodePort on port 30080
