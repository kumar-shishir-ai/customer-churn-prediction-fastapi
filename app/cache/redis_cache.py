import json
import redis
from app.core.config import settings

redis_client = redis.Redis.from_url(settings.REDIS_URL)

def get_cached_prediction(key: str):
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
    except redis.exceptions.ConnectionError:
        pass
    return None

def set_cached_prediction(key: str, value: dict, expity: int = 3600):
    try:
        redis_client.setex(key, expity, json.dumps(value))
    except redis.exceptions.ConnectionError:
        pass