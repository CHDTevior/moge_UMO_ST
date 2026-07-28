"""Disabled legacy HumanML3D evaluator wrapper.

The active branch is standardized on the MotionStreamer 272D evaluator.  This
module keeps the old class names importable so stale code fails with a clear
message instead of silently using the removed evaluator path.
"""


class EvaluatorModelWrapper:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The legacy HumanML3D evaluator wrapper has been disabled. "
            "Use the MotionStreamer272 evaluator path for FID/Top3."
        )


class EvaluatorWrapper:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "The legacy HumanML3D evaluator wrapper has been disabled. "
            "Use the MotionStreamer272 evaluator path for FID/Top3."
        )
