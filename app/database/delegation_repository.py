"""复用运行 Repository 的租约和事务，不维护第二套业务提交器。"""
import json
from sqlalchemy import select, update
from app.database.models import ResearchDelegationModel as Delegation
from app.database.models import ResearchInvocationModel as Invocation
from app.database.models import ResearchWorkVersionModel as Work
from app.database.models import ResearchEventModel as Event
from app.database.runtime_repository import RuntimeConflict, dumps
from app.schemas.delegation_schema import ActionInvocation, DelegationGrant


class DelegationRepository:
    def __init__(self, runtime):
        self.runtime = runtime
        self.db = runtime.db

    def version(self):
        row = self.db.get(Work, self.runtime.session_id, populate_existing=True)
        return row.version if row else 0

    def create(self, grant):
        if grant.session_id != self.runtime.session_id:
            raise ValueError("grant belongs to another session")
        with self.runtime.transaction():
            self.runtime._owned_row_for_write(live=True)
            if self.active() is not None:
                raise RuntimeConflict('another delegation is already active')
            if self.db.get(Delegation, grant.delegation_id):
                raise ValueError("delegation already exists")
            if self.db.get(Work, grant.session_id) is None:
                self.db.add(Work(session_id=grant.session_id, version=0))
            self.db.add(Delegation(delegation_id=grant.delegation_id, session_id=grant.session_id,
                                   grant_json=grant.model_dump_json(), status="running"))

    def active(self):
        rows = list(self.db.scalars(select(Delegation).where(
            Delegation.session_id == self.runtime.session_id, Delegation.status == 'running')
            .execution_options(populate_existing=True)))
        if len(rows) > 1:
            raise RuntimeConflict('multiple active delegations require explicit recovery')
        return rows[0] if rows else None

    def read(self, delegation_id):
        row = self.db.get(Delegation, delegation_id, populate_existing=True)
        if row is None or row.session_id != self.runtime.session_id:
            raise ValueError("unknown delegation")
        return row, DelegationGrant.model_validate_json(row.grant_json)

    def prepare(self, invocation):
        with self.runtime.transaction():
            self.runtime._owned_row_for_write(live=True)
            row, grant = self.read(invocation.delegation_id)
            existing = self.db.get(Invocation, invocation.invocation_id)
            if existing:
                if ActionInvocation.model_validate_json(existing.invocation_json) != invocation:
                    raise RuntimeConflict("invocation identity changed")
                return json.loads(existing.result_json) if existing.status == "committed" else None
            if row.status != "running" or invocation.step_index != row.next_step:
                raise RuntimeConflict("delegation cursor changed")
            if row.next_step >= grant.max_steps or invocation.expected_version != self.version():
                raise RuntimeConflict("invocation version or budget changed")
            occupied = self.db.scalar(select(Invocation).where(
                Invocation.delegation_id == invocation.delegation_id, Invocation.step_index == invocation.step_index))
            if occupied is not None:
                raise RuntimeConflict("step already has a stable invocation")
            self.db.add(Invocation(invocation_id=invocation.invocation_id, delegation_id=invocation.delegation_id,
                step_index=invocation.step_index, invocation_json=invocation.model_dump_json()))
        return None

    def committed_result(self, invocation):
        self.runtime.check()
        self.read(invocation.delegation_id)
        saved = self.db.get(Invocation, invocation.invocation_id, populate_existing=True)
        if saved is None:
            return None
        if ActionInvocation.model_validate_json(saved.invocation_json) != invocation:
            raise RuntimeConflict("invocation identity changed")
        return json.loads(saved.result_json) if saved.status == "committed" else None

    def bind_task(self, invocation, task):
        # WHY: 由 begin_task 的事务调用，与实际尝试记录共同落库。
        row, grant = self.read(invocation.delegation_id)
        saved = self.db.get(Invocation, invocation.invocation_id)
        if (saved is None or saved.status == "committed" or row.status != "running"
                or invocation.expected_version != self.version() or invocation.grant_revision != grant.revision
                or task.paper_scope != list(grant.scope.get('paper_ids') or [])):
            raise RuntimeConflict("invocation cannot start")
        saved.task_id = task.task_id
        saved.status = "running"

    def revoke(self, delegation_id):
        with self.runtime.transaction():
            self.runtime._owned_row_for_write(live=True)
            row, _ = self.read(delegation_id)
            row.status = "cancelled"

    def commit(self, invocation, task, result):
        row, grant = self.read(invocation.delegation_id)
        saved = self.db.get(Invocation, invocation.invocation_id)
        if (saved is None or saved.task_id != task.task_id or saved.status != "running"
                or row.status != "running" or row.next_step != invocation.step_index
                or invocation.grant_revision != grant.revision or task.role != grant.role
                or invocation.action not in grant.allowed_actions):
            raise RuntimeConflict("invocation no longer active")
        changed = self.db.execute(update(Work).where(Work.session_id == self.runtime.session_id,
            Work.version == invocation.expected_version).values(version=Work.version + 1))
        if changed.rowcount != 1:
            raise RuntimeConflict("business version CAS rejected")
        saved.status = "committed"
        saved.result_json = dumps(result.model_dump(mode="json"))
        row.next_step += 1
        # WHY: durable observation 在结果中；消费游标待 Runner 实际读取后独立推进。

    def consume(self, delegation_id, cursor):
        with self.runtime.transaction():
            self.runtime._owned_row_for_write(live=True)
            row, _ = self.read(delegation_id)
            if not row.observation_cursor <= cursor <= row.next_step:
                raise ValueError("invalid observation cursor")
            row.observation_cursor = cursor

    def invocations(self, delegation_id):
        self.read(delegation_id)
        return list(self.db.scalars(select(Invocation).where(Invocation.delegation_id == delegation_id)
                                   .order_by(Invocation.step_index).execution_options(populate_existing=True)))

    def finish(self, result):
        with self.runtime.transaction():
            self.runtime._owned_row_for_write(live=True)
            row, _ = self.read(result.delegation_id)
            if row.status != 'running':
                raise RuntimeConflict('delegation already terminated')
            row.status = result.status
            self.runtime.event('delegation-result:' + row.delegation_id, 'delegation_completed',
                               result.model_dump(mode='json'))

    def completed_result(self, delegation_id):
        row, _ = self.read(delegation_id)
        event = self.db.scalar(select(Event).where(Event.session_id == self.runtime.session_id,
            Event.event_key == 'delegation-result:' + delegation_id))
        return json.loads(event.payload_json) if row.status == 'completed' and event else None
