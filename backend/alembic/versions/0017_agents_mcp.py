"""agents + mcp servers tables, tool_call_stats, request_stats extensions

Revision ID: 0017
Revises: 0016
Create Date: 2026-04-24

Plan: docs/tasks/agents-mcp-plan.md — Fase 1 "Schema + registry".
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0017"
down_revision: str | None = "0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ── mcp_servers ────────────────────────────────────────────
    op.create_table(
        "mcp_servers",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column("alias", sa.String(128), nullable=False, unique=True),
        sa.Column("display_name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("transport", sa.String(16), nullable=False),  # http | sse | stdio

        # HTTP / SSE
        sa.Column("url", sa.Text(), nullable=True),

        # stdio
        sa.Column("command", sa.Text(), nullable=True),
        sa.Column("args", postgresql.JSONB(), nullable=True),
        # env is stored as Fernet ciphertext (TEXT) because it may contain secrets.
        sa.Column("env_encrypted", sa.Text(), nullable=True),

        # Auth
        sa.Column("auth_type", sa.String(32), nullable=False, server_default="none"),
        # Fernet ciphertext of the serialized auth payload (header name + value, etc.)
        sa.Column("auth_value_encrypted", sa.Text(), nullable=True),
        sa.Column("oauth_config", postgresql.JSONB(), nullable=True),

        # ACL
        sa.Column("allowed_tools", postgresql.JSONB(), nullable=True),
        sa.Column(
            "group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="SET NULL"),
            nullable=True,
        ),

        # Runtime cache
        sa.Column("tools_cache", postgresql.JSONB(), nullable=True),
        sa.Column(
            "tools_cache_updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("health_status", sa.String(32), nullable=False, server_default="unknown"),

        # Audit
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_mcp_servers_alias", "mcp_servers", ["alias"])
    op.create_index("ix_mcp_servers_group_id", "mcp_servers", ["group_id"])
    op.create_index("ix_mcp_servers_transport", "mcp_servers", ["transport"])

    # ── agents ─────────────────────────────────────────────────
    op.create_table(
        "agents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column("slug", sa.String(128), nullable=False, unique=True),
        sa.Column("display_name", sa.String(256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "base_model_id",
            sa.String(512),
            sa.ForeignKey("model_configs.model_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "profile_id",
            sa.String(512),
            sa.ForeignKey("model_profiles.profile_id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("system_prompt", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "tool_choice_default",
            sa.String(16),
            nullable=False,
            server_default="auto",
        ),
        sa.Column(
            "max_tool_hops",
            sa.Integer(),
            nullable=False,
            server_default="8",
        ),
        sa.Column(
            "tool_timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default="60",
        ),
        sa.Column(
            "require_approval",
            sa.String(16),
            nullable=False,
            server_default="never",
        ),
        sa.Column("request_defaults", postgresql.JSONB(), nullable=True),
        sa.Column(
            "group_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("groups.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(base_model_id IS NOT NULL) <> (profile_id IS NOT NULL)",
            name="ck_agents_exactly_one_base",
        ),
    )
    op.create_index("ix_agents_slug", "agents", ["slug"])
    op.create_index("ix_agents_group_id", "agents", ["group_id"])

    # ── agent_mcp_servers (many-to-many) ──────────────────────
    op.create_table(
        "agent_mcp_servers",
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "mcp_server_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("mcp_servers.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("allowed_tools", postgresql.JSONB(), nullable=True),
    )
    op.create_index(
        "ix_agent_mcp_servers_mcp_server_id",
        "agent_mcp_servers",
        ["mcp_server_id"],
    )

    # ── tool_call_stats ────────────────────────────────────────
    op.create_table(
        "tool_call_stats",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "request_stat_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("request_stats.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("mcp_server_alias", sa.String(128), nullable=False),
        sa.Column("tool_name", sa.String(256), nullable=False),
        sa.Column("tool_args_redacted", postgresql.JSONB(), nullable=True),
        sa.Column("result_summary", sa.Text(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False, server_default="ok"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("hop_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_tool_call_stats_request_stat_id",
        "tool_call_stats",
        ["request_stat_id"],
    )
    op.create_index("ix_tool_call_stats_agent_id", "tool_call_stats", ["agent_id"])
    op.create_index(
        "ix_tool_call_stats_alias_tool",
        "tool_call_stats",
        ["mcp_server_alias", "tool_name"],
    )
    op.create_index("ix_tool_call_stats_created_at", "tool_call_stats", ["created_at"])

    # ── request_stats: agent_id + parent_request_id ───────────
    op.add_column(
        "request_stats",
        sa.Column(
            "agent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("agents.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "request_stats",
        sa.Column(
            "parent_request_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("request_stats.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.create_index("ix_request_stats_agent_id", "request_stats", ["agent_id"])
    op.create_index(
        "ix_request_stats_parent_request_id",
        "request_stats",
        ["parent_request_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_request_stats_parent_request_id", table_name="request_stats")
    op.drop_index("ix_request_stats_agent_id", table_name="request_stats")
    op.drop_column("request_stats", "parent_request_id")
    op.drop_column("request_stats", "agent_id")

    op.drop_index("ix_tool_call_stats_created_at", table_name="tool_call_stats")
    op.drop_index("ix_tool_call_stats_alias_tool", table_name="tool_call_stats")
    op.drop_index("ix_tool_call_stats_agent_id", table_name="tool_call_stats")
    op.drop_index("ix_tool_call_stats_request_stat_id", table_name="tool_call_stats")
    op.drop_table("tool_call_stats")

    op.drop_index(
        "ix_agent_mcp_servers_mcp_server_id", table_name="agent_mcp_servers"
    )
    op.drop_table("agent_mcp_servers")

    op.drop_index("ix_agents_group_id", table_name="agents")
    op.drop_index("ix_agents_slug", table_name="agents")
    op.drop_table("agents")

    op.drop_index("ix_mcp_servers_transport", table_name="mcp_servers")
    op.drop_index("ix_mcp_servers_group_id", table_name="mcp_servers")
    op.drop_index("ix_mcp_servers_alias", table_name="mcp_servers")
    op.drop_table("mcp_servers")
