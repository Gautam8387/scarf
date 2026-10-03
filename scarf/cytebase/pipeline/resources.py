"""Fixed resource steps for dataset workers and safe import refusals."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessResources:
    cpu: int
    memoryMiB: int
    memBudget: str


PROCESS_RESOURCES = (
    ProcessResources(cpu=4, memoryMiB=16_384, memBudget="12G"),
    ProcessResources(cpu=8, memoryMiB=32_768, memBudget="24G"),
    ProcessResources(cpu=16, memoryMiB=65_536, memBudget="48G"),
)


class ImportMemoryRefusal(MemoryError):
    """Default-layout admission refused before creating the local store."""
