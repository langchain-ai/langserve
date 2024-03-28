from typing import Dict, List
from uuid import UUID

from langchain_core.tracers import BaseTracer
from langsmith.schemas import Run


class FakeTracer(BaseTracer):
    """Fake tracer that records LangChain execution.

    It replaces run ids with deterministic UUIDs."""

    def __init__(self) -> None:
        """Initialize the tracer."""
        super().__init__()
        self.runs: List[Run] = []
        self.uuids_map: Dict[UUID, UUID] = {}
        self.uuids_generator = (
            UUID(f"00000000-0000-4000-8000-{i:012}", version=4) for i in range(10_000)
        )

    def _replace_uuid(self, uuid: UUID) -> UUID:
        """Replace a UUID with a deterministic one."""
        if uuid not in self.uuids_map:
            self.uuids_map[uuid] = next(self.uuids_generator)
        return self.uuids_map[uuid]

    def _copy_run(self, run: Run) -> Run:
        """Copy a run, replacing UUIDs."""
        return run.copy(
            update={
                "id": self._replace_uuid(run.id),
                "parent_run_id": self.uuids_map[run.parent_run_id]
                if run.parent_run_id
                else None,
                "child_runs": [self._copy_run(child) for child in run.child_runs],
                "execution_order": None,
                "child_execution_order": None,
            }
        )

    def _persist_run(self, run: Run) -> None:
        """Persist a run."""
        self.runs.append(self._copy_run(run))
