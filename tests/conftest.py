import pytest

from jaswolf.config import JaswolfSettings
from jaswolf.service import MemoryService


@pytest.fixture
def settings(tmp_path) -> JaswolfSettings:
    return JaswolfSettings(
        database_url=f"sqlite:///{tmp_path}/jaswolf_test.db",
        embedding_provider="hash",
        embedding_dim=384,
        api_keys="",
        dev_open_mode=True,  # API tests run without auth unless they opt in
        sweep_interval_seconds=3600,
        log_level="WARNING",
    )


@pytest.fixture
async def service(settings) -> MemoryService:
    svc = await MemoryService.create(settings)
    yield svc
    await svc.close()
