import os
from contextlib import contextmanager
from time import perf_counter

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import MODEL_NAME
from runtime_context import current_run, current_session

_provider = None


def setup_tracing(app=None):
    global _provider
    if _provider is None:
        _provider = TracerProvider(resource=Resource.create({"service.name": "zerocode-backend"}))
        if os.getenv("OTEL_TRACES_EXPORTER", "otlp") != "none":
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            exporter = OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
                                        insecure=True, timeout=3)
            _provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_provider)
    if app is not None:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_provider)
    return _provider


@contextmanager
def traced(name, input_shape=""):
    setup_tracing()
    start = perf_counter()
    with trace.get_tracer("zerocode").start_as_current_span(name) as span:
        sid = current_session.get()
        span.set_attributes({"session_id": sid, "run_id": current_run.get(),
                             "input_shape": str(input_shape), "model_used": MODEL_NAME})
        if sid != "legacy":
            from blackboard import Blackboard
            Blackboard(sid).add_trace(f"{span.get_span_context().trace_id:032x}")
        try:
            yield span
        finally:
            span.set_attribute("duration_ms", (perf_counter() - start) * 1000)


def trace_link(session_id):
    from blackboard import Blackboard
    ids = Blackboard(session_id).get_full_state()["trace_ids"]
    base = os.getenv("JAEGER_UI_URL", "http://localhost:16686").rstrip("/")
    return {"session_id": session_id, "trace_ids": ids,
            "url": f"{base}/trace/{ids[-1]}" if ids else None,
            "urls": [f"{base}/trace/{trace_id}" for trace_id in ids]}
