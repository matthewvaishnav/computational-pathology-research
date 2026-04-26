"""Initial database schema for FL system

Revision ID: 001
Revises: 
Create Date: 2026-04-25 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create clients table
    op.create_table('clients',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('client_id', sa.String(length=255), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('organization', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('last_seen', sa.DateTime(), nullable=True),
        sa.Column('max_batch_size', sa.Integer(), nullable=True),
        sa.Column('supported_algorithms', sa.JSON(), nullable=True),
        sa.Column('privacy_budget_epsilon', sa.Float(), nullable=False),
        sa.Column('privacy_budget_delta', sa.Float(), nullable=False),
        sa.Column('max_epsilon', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_clients_client_id'), 'clients', ['client_id'], unique=True)

    # Create training_rounds table
    op.create_table('training_rounds',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('round_id', sa.Integer(), nullable=False),
        sa.Column('algorithm', sa.String(length=100), nullable=False),
        sa.Column('min_clients', sa.Integer(), nullable=False),
        sa.Column('max_clients', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('participants', sa.JSON(), nullable=True),
        sa.Column('aggregated_metrics', sa.JSON(), nullable=True),
        sa.Column('model_version', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_training_rounds_round_id'), 'training_rounds', ['round_id'], unique=False)

    # Create client_updates table
    op.create_table('client_updates',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('round_id', sa.Integer(), nullable=False),
        sa.Column('client_id', sa.String(length=255), nullable=False),
        sa.Column('dataset_size', sa.Integer(), nullable=False),
        sa.Column('training_time_seconds', sa.Float(), nullable=False),
        sa.Column('privacy_epsilon', sa.Float(), nullable=False),
        sa.Column('privacy_delta', sa.Float(), nullable=False),
        sa.Column('model_update_path', sa.String(length=500), nullable=False),
        sa.Column('model_update_hash', sa.String(length=64), nullable=False),
        sa.Column('is_valid', sa.Boolean(), nullable=False),
        sa.Column('validation_errors', sa.JSON(), nullable=True),
        sa.Column('is_byzantine', sa.Boolean(), nullable=False),
        sa.Column('byzantine_score', sa.Float(), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=False),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_client_updates_round_id'), 'client_updates', ['round_id'], unique=False)
    op.create_index(op.f('ix_client_updates_client_id'), 'client_updates', ['client_id'], unique=False)

    # Create model_checkpoints table
    op.create_table('model_checkpoints',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('round_id', sa.Integer(), nullable=False),
        sa.Column('checkpoint_path', sa.String(length=500), nullable=False),
        sa.Column('checkpoint_hash', sa.String(length=64), nullable=False),
        sa.Column('size_bytes', sa.Integer(), nullable=False),
        sa.Column('validation_metrics', sa.JSON(), nullable=True),
        sa.Column('training_metrics', sa.JSON(), nullable=True),
        sa.Column('contributors', sa.JSON(), nullable=True),
        sa.Column('algorithm_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_checkpoints_version'), 'model_checkpoints', ['version'], unique=True)

    # Create audit_logs table
    op.create_table('audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('event_category', sa.String(length=50), nullable=False),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('client_id', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('resource_type', sa.String(length=100), nullable=True),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('additional_data', sa.JSON(), nullable=True),
        sa.Column('success', sa.Boolean(), nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audit_logs_event_type'), 'audit_logs', ['event_type'], unique=False)
    op.create_index(op.f('ix_audit_logs_event_category'), 'audit_logs', ['event_category'], unique=False)
    op.create_index(op.f('ix_audit_logs_user_id'), 'audit_logs', ['user_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_client_id'), 'audit_logs', ['client_id'], unique=False)
    op.create_index(op.f('ix_audit_logs_timestamp'), 'audit_logs', ['timestamp'], unique=False)

    # Create system_metrics table
    op.create_table('system_metrics',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metric_name', sa.String(length=100), nullable=False),
        sa.Column('metric_value', sa.Float(), nullable=False),
        sa.Column('metric_unit', sa.String(length=50), nullable=True),
        sa.Column('component', sa.String(length=100), nullable=False),
        sa.Column('instance_id', sa.String(length=255), nullable=True),
        sa.Column('labels', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_metrics_metric_name'), 'system_metrics', ['metric_name'], unique=False)
    op.create_index(op.f('ix_system_metrics_component'), 'system_metrics', ['component'], unique=False)
    op.create_index(op.f('ix_system_metrics_timestamp'), 'system_metrics', ['timestamp'], unique=False)


def downgrade() -> None:
    op.drop_table('system_metrics')
    op.drop_table('audit_logs')
    op.drop_table('model_checkpoints')
    op.drop_table('client_updates')
    op.drop_table('training_rounds')
    op.drop_table('clients')