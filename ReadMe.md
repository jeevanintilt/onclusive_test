# PUBHEALTH Claim Veracity Classifier

This project fine-tunes a Longformer model to classify health-related claims into the following categories: true, false, unproven, and mixture. The solution includes data preprocessing, model training, evaluation, and deployment via a REST API.

## setting up virtual Environment
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install -r requirements.txt

## Usage
### Download Data : 
python ingest.py --output_dir ./data

### Preprocess Data:
python prepare.py --input_dir ./data --output_dir ./processed_data --model_name nbroad/longformer-base-health-fact

### Train Model:
python train.py --data_dir ./processed_data --output_dir ./fine_tuned_model --epochs 3 --batch_size 8 --lr 3e-5

### Serve Model:
python serve.py
API Docs: http://127.0.0.1:8000/docs

### Run unit tests:
pytest test_serve.py

## Deployment
### Running as Docker
docker build -t health-claim-serve:v1 .
docker tag health-claim-serve:v1 <your-registry>/health-claim-serve:v1
docker push <your-registry>/health-claim-serve:v1

### Run container
docker run -p 8000:8000 health-claim-serve

### Running in AKS:
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl get pods
kubectl get services