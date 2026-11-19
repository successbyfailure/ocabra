"""Borrar un agente lo quita de los ``subagent_slugs`` de los demás (Postgres real)."""

from __future__ import annotations

import uuid

import sqlalchemy as sa

from ocabra.api.internal.agents import delete_agent
from ocabra.database import AsyncSessionLocal
from ocabra.db.agents import Agent
from ocabra.db.model_config import LogicalModel, ModelConfig
from tests.conftest import requires_db


@requires_db
async def test_delete_drops_references_in_the_same_transaction() -> None:
    tag = f"zz-agentdel-{uuid.uuid4().hex[:6]}"
    model_id = f"vllm/{tag}"
    child, parent, other = f"{tag}-child", f"{tag}-parent", f"{tag}-other"

    async def cleanup() -> None:
        async with AsyncSessionLocal() as session:
            await session.execute(sa.delete(Agent).where(Agent.slug.like(f"{tag}-%")))
            await session.execute(sa.delete(ModelConfig).where(ModelConfig.model_key == tag))
            await session.execute(sa.delete(LogicalModel).where(LogicalModel.model_key == tag))
            await session.commit()

    async with AsyncSessionLocal() as session:
        session.add(LogicalModel(model_key=tag, default_backend="vllm"))
        await session.flush()
        session.add(ModelConfig(model_id=model_id, model_key=tag, backend_type="vllm"))
        await session.flush()
        for slug, subs in ((child, []), (other, []), (parent, [other, child])):
            session.add(
                Agent(slug=slug, display_name=slug, base_model_id=model_id, subagent_slugs=subs)
            )
        await session.commit()
    try:
        result = await delete_agent(child, _user=None)

        assert result == {"ok": True, "slug": child, "detached_from": [parent]}
        async with AsyncSessionLocal() as session:
            rows = dict(
                (
                    await session.execute(
                        sa.select(Agent.slug, Agent.subagent_slugs).where(
                            Agent.slug.like(f"{tag}-%")
                        )
                    )
                ).all()
            )
        assert rows == {parent: [other], other: []}
    finally:
        await cleanup()
