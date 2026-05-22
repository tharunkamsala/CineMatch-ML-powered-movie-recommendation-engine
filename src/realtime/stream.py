"""
Rating ingestion stream — Kafka backend with in-memory queue fallback.
Producers emit new ratings; consumers trigger online model updates.
"""
import json
import queue
import threading
import time
from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class RatingEvent:
    user_id:   int
    movie_id:  int
    rating:    float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class RatingStream:
    """
    Kafka-backed stream with graceful in-memory fallback.
    Usage:
        stream = RatingStream(kafka_url="localhost:9092", topic="ratings")
        stream.produce(RatingEvent(user_id=1, movie_id=42, rating=4.5))
        stream.consume(callback=my_handler)
    """

    TOPIC = "cinемatch.ratings"

    def __init__(self, kafka_url: Optional[str] = None, topic: str = "cinematch.ratings"):
        self.topic      = topic
        self._producer  = None
        self._consumer  = None
        self._local_q   = queue.Queue()
        self._kafka_ok  = False
        self._total_produced = 0
        self._total_consumed = 0

        if kafka_url:
            try:
                from kafka import KafkaProducer, KafkaConsumer
                self._producer = KafkaProducer(
                    bootstrap_servers=kafka_url,
                    value_serializer=lambda v: json.dumps(v).encode(),
                    request_timeout_ms=3000,
                )
                self._consumer_cls = KafkaConsumer
                self._kafka_url    = kafka_url
                self._kafka_ok     = True
                print(f"Stream: connected to Kafka at {kafka_url}")
            except Exception as e:
                print(f"Stream: Kafka unavailable ({e}), using in-memory queue")
        else:
            print("Stream: no Kafka URL provided, using in-memory queue")

    def produce(self, event: RatingEvent) -> None:
        self._total_produced += 1
        if self._kafka_ok:
            self._producer.send(self.topic, asdict(event))
        else:
            self._local_q.put(asdict(event))

    def consume(self, callback: Callable[[RatingEvent], None], block: bool = False) -> None:
        """
        Start consuming events. If block=True, runs in the calling thread.
        If block=False, spawns a daemon thread.
        """
        def _run():
            if self._kafka_ok:
                self._consume_kafka(callback)
            else:
                self._consume_local(callback)

        if block:
            _run()
        else:
            t = threading.Thread(target=_run, daemon=True)
            t.start()

    def _consume_kafka(self, callback: Callable):
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self._kafka_url,
            value_deserializer=lambda v: json.loads(v.decode()),
            auto_offset_reset="latest",
        )
        for msg in consumer:
            event = RatingEvent(**msg.value)
            callback(event)
            self._total_consumed += 1

    def _consume_local(self, callback: Callable):
        while True:
            try:
                data  = self._local_q.get(timeout=1.0)
                event = RatingEvent(**data)
                callback(event)
                self._total_consumed += 1
            except queue.Empty:
                continue

    def stats(self) -> dict:
        return {
            "backend":  "kafka" if self._kafka_ok else "in-memory",
            "produced": self._total_produced,
            "consumed": self._total_consumed,
            "queue_size": self._local_q.qsize() if not self._kafka_ok else "n/a",
        }
