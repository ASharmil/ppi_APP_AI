from celery import Celery
from app.core.config import settings

celery_app = Celery(
    "ppi_backend",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=[
        "app.tasks.tasks_training",
        "app.tasks.tasks_drug_sync",
        "app.tasks.tasks_prediction"
    ]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_routes={
        "app.tasks.tasks_training.*": {"queue": "training"},
        "app.tasks.tasks_drug_sync.*": {"queue": "sync"},
        "app.tasks.tasks_prediction.*": {"queue": "prediction"}
    }
)
