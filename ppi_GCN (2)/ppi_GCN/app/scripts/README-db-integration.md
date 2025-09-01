# Python ↔ API integration

1) Create dataset
- POST /api/datasets with JSON: { "name": "My dataset", "csvText": "<protein_a,protein_b,label... csv>" }
- Response: { id }

2) Start run
- POST /api/runs with JSON: { "dataset_id": "<id>", "notes": "baseline gcn" }
- Copy the hint commands from the response (contains dataset download URL, train/infer examples).

3) Download CSV locally for training
- curl -sSL "<BASE_URL>/api/datasets/<id>/download" -o data/train.csv

4) Train with your existing script
- python -m scripts.train_gcn_v1 --csv data/train.csv --save-dir artifacts/<run_id>

5) Inference (dynamic pairs or CSV) with your script
- python -m scripts.infer_gcn_v1 --csv data/dynamic_eval.csv --artifacts artifacts/<run_id>/artifacts.json --model artifacts/<run_id>/gcn_lp.pt --out outputs/<run_id>_scores.csv

6) Upload metrics
- POST /api/runs/<run_id>/metrics with JSON body: { "status": "completed", "metrics": { "val_best_acc": 0.88, "test_loss": 0.42, "test_acc": 0.85 } }

7) Upload predictions
- Convert your scores.csv into JSON array:
  [
    { "run_id": "<run_id>", "protein_a": "P12345", "protein_b": "Q9Y2Z2", "probability": 0.91, "drug_targets_overlap": false },
    ...
  ]
- POST /api/predictions with that JSON. Then fetch via GET /api/predictions?run_id=<run_id>.
