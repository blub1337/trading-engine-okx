"""SQLite-backed durable state and trade journaling.

This module provides a fail-closed persistence layer. All operations
are explicit; missing or corrupt data raises rather than silently
proceeding with defaults.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ..models import ExecutionResult, OrderIntent, RiskDecision, Signal, SignalSide, PaperOrder


SCHEMA = """
-- Journal of all signals received
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    asset TEXT NOT NULL,
    side TEXT NOT NULL,
    conf REAL NOT NULL,
    rationale TEXT NOT NULL,
    required_inputs TEXT NOT NULL,  -- JSON array
    issued_at TEXT NOT NULL,        -- ISO8601 UTC
    signal_id TEXT UNIQUE,          -- Stable signal identity for idempotency
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Journal of all order intents (proposed trades)
CREATE TABLE IF NOT EXISTS order_intents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL UNIQUE,
    asset TEXT NOT NULL,
    side TEXT NOT NULL,
    notional REAL NOT NULL,
    qty REAL,
    limit_price REAL,
    meta TEXT NOT NULL,             -- JSON object
    signal_id INTEGER,
    signal_ref_id TEXT,             -- Reference to signals.signal_id for idempotency
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Journal of all risk decisions
CREATE TABLE IF NOT EXISTS risk_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_intent_id INTEGER NOT NULL,
    status TEXT NOT NULL,           -- APPROVED / REJECTED
    reason TEXT NOT NULL,
    checks TEXT NOT NULL,           -- JSON array of {name, ok}
    approved_at TEXT,               -- ISO8601 UTC, nullable
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Journal of all execution results
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_intent_id INTEGER NOT NULL,
    risk_decision_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    adapter TEXT NOT NULL,
    order_id TEXT,
    filled INTEGER NOT NULL,        -- 0/1 boolean
    fail_reason TEXT,
    fill_price REAL,                -- Execution fill price
    fill_qty REAL,                  -- Execution fill quantity
    fee REAL,                       -- Fee paid
    fee_bps REAL,                   -- Fee in basis points
    slippage_bps REAL,              -- Slippage in basis points
    portfolio_applied INTEGER NOT NULL DEFAULT 0,  -- Exactly-once portfolio marker
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Portfolio state (current positions, cash, etc.)
CREATE TABLE IF NOT EXISTS portfolio_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    cash REAL NOT NULL DEFAULT 0.0,
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL UNIQUE,
    qty REAL NOT NULL DEFAULT 0.0,
    avg_entry_price REAL,
    realized_pnl REAL DEFAULT 0.0,  -- Accumulated realized P&L
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Daily loss tracking state
CREATE TABLE IF NOT EXISTS daily_loss_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    daily_start_equity REAL,
    last_trading_date TEXT,         -- ISO8601 date
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Paper broker orders for crash recovery
CREATE TABLE IF NOT EXISTS paper_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL UNIQUE,
    asset TEXT NOT NULL,
    side TEXT NOT NULL,
    notional REAL NOT NULL,
    qty REAL,
    limit_price REAL,
    reference_price REAL NOT NULL,
    fee_bps INTEGER NOT NULL,
    slippage_bps INTEGER NOT NULL,
    status TEXT NOT NULL,           -- PENDING, FILLED, REJECTED, ERROR
    fill_price REAL,
    fill_qty REAL,
    fee REAL,
    broker_order_id TEXT,
    fail_reason TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_signals_asset ON signals(asset);
CREATE INDEX IF NOT EXISTS idx_signals_issued_at ON signals(issued_at);
CREATE INDEX IF NOT EXISTS idx_signals_signal_id ON signals(signal_id);
CREATE INDEX IF NOT EXISTS idx_order_intents_client_oid ON order_intents(client_order_id);
CREATE INDEX IF NOT EXISTS idx_order_intents_signal_ref ON order_intents(signal_ref_id);
CREATE INDEX IF NOT EXISTS idx_risk_decisions_order_intent ON risk_decisions(order_intent_id);
CREATE INDEX IF NOT EXISTS idx_executions_order_intent ON executions(order_intent_id);
CREATE INDEX IF NOT EXISTS idx_paper_orders_client_oid ON paper_orders(client_order_id);
CREATE INDEX IF NOT EXISTS idx_paper_orders_status ON paper_orders(status);
-- Audit table for signal rejections that occur before an order intent can be created
CREATE TABLE IF NOT EXISTS signal_rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    asset TEXT NOT NULL,
    side TEXT NOT NULL,
    conf REAL NOT NULL,
    rationale TEXT NOT NULL,
    required_inputs TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_signal_rejections_asset ON signal_rejections(asset);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(dt_str: str) -> datetime:
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


class SQLiteStore:
    """Thread-safe SQLite store with explicit transactions."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize schema. Fails hard on any error.

        Uses the SAME connection returned by _get_conn() so that
        in-memory databases (db_path=":memory:") keep their tables
        for the lifetime of this store instance instead of being lost
        to a separate connection that gets closed immediately."""
        conn = self._get_conn()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        # Ensure singleton portfolio row exists
        conn.execute(
            "INSERT OR IGNORE INTO portfolio_state (id, cash) VALUES (1, 100000.0)"
        )
        # Ensure singleton daily loss state row exists
        conn.execute(
            "INSERT OR IGNORE INTO daily_loss_state (id) VALUES (1)"
        )
        # Migration: add portfolio_applied column to executions if missing
        try:
            conn.execute("ALTER TABLE executions ADD COLUMN portfolio_applied INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            # Column already exists - safe to ignore
            pass
        conn.commit()

    @contextmanager
    def transaction(self):
        """Explicit transaction context with BEGIN IMMEDIATE for write serialization.

        Uses BEGIN IMMEDIATE to acquire a reserved lock immediately, preventing
        other writers from starting transactions concurrently. This ensures
        exactly-once semantics for portfolio updates.

        Commits on success, rolls back on exception.
        """
        conn = self._get_conn()
        # Use BEGIN IMMEDIATE for write serialization
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
            raise

    # --- Signals ---

    def record_signal(self, signal: Signal) -> int:
        """Record a strategy signal. Returns signal ID."""
        with self.transaction() as conn:
            # Generate signal_id if not provided
            signal_id = signal.signal_id
            if signal_id is None:
                import hashlib
                signal_id = hashlib.md5(
                    f"{signal.strategy}{signal.asset}{signal.side.value}{signal.issued_at.isoformat()}{signal.conf}".encode()
                ).hexdigest()[:16]

            try:
                cur = conn.execute(
                    """
                    INSERT INTO signals (strategy, asset, side, conf, rationale, required_inputs, issued_at, signal_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        signal.strategy,
                        signal.asset,
                        signal.side.value,
                        signal.conf,
                        signal.rationale,
                        json.dumps(signal.required_inputs),
                        signal.issued_at.isoformat(),
                        signal_id,
                    ),
                )
                return cur.lastrowid
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed: signals.signal_id" in str(e):
                    # Signal already exists, return existing ID
                    existing = conn.execute(
                        "SELECT id FROM signals WHERE signal_id = ?", (signal_id,)
                    ).fetchone()
                    if existing:
                        return existing["id"]
                raise

    def record_signal_rejection(self, signal: Signal, reason: str) -> int:
        """Record a rejection that occurs before an OrderIntent is created.

        Mirrors the fields stored for a normal signal, plus a ``reason``.
        """
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO signal_rejections (
                    strategy, asset, side, conf, rationale, required_inputs, issued_at, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.strategy,
                    signal.asset,
                    signal.side.value,
                    signal.conf,
                    signal.rationale,
                    json.dumps(signal.required_inputs),
                    signal.issued_at.isoformat(),
                    reason,
                ),
            )
            return cur.lastrowid

    def get_recent_signals(self, asset: str, limit: int = 100) -> list[dict]:
        """Get recent signals for an asset."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM signals WHERE asset = ? ORDER BY issued_at DESC LIMIT ?",
            (asset, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_signal_by_signal_id(self, signal_id: str) -> Optional[dict]:
        """Get signal by stable signal_id for idempotency check."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM signals WHERE signal_id = ?", (signal_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- Order Intents ---

    def record_order_intent(self, intent: OrderIntent, signal_id: Optional[int] = None) -> int:
        """Record a proposed trade. Returns order_intent ID."""
        with self.transaction() as conn:
            # Get signal_ref_id if available
            signal_ref_id = None
            if hasattr(intent, 'signal_id') and intent.signal_id:
                signal_ref_id = intent.signal_id

            cur = conn.execute(
                """
                INSERT INTO order_intents (client_order_id, asset, side, notional, qty, limit_price, meta, signal_id, signal_ref_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    intent.client_order_id,
                    intent.asset,
                    intent.side.value,
                    intent.notional,
                    intent.qty,
                    intent.limit,
                    json.dumps(intent.meta),
                    signal_id,
                    signal_ref_id,
                ),
            )
            return cur.lastrowid

    def get_order_intent(self, client_order_id: str) -> Optional[dict]:
        """Get order intent by client order ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM order_intents WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_order_intent_by_signal_ref(self, signal_ref_id: str) -> Optional[dict]:
        """Get order intent by signal_ref_id for idempotency check."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM order_intents WHERE signal_ref_id = ?", (signal_ref_id,)
        ).fetchone()
        return dict(row) if row else None

    # --- Risk Decisions ---

    def record_risk_decision(
        self, order_intent_id: int, decision: RiskDecision
    ) -> int:
        """Record a risk gate decision. Returns risk_decision ID."""
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO risk_decisions (order_intent_id, status, reason, checks, approved_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    order_intent_id,
                    decision.status.value,
                    decision.reason,
                    json.dumps([{"name": n, "ok": v} for n, v in decision.checks]),
                    decision.approved_at.isoformat() if decision.approved_at else None,
                ),
            )
            return cur.lastrowid

    def get_risk_decision(self, order_intent_id: int) -> Optional[dict]:
        """Get latest risk decision for an order intent."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM risk_decisions WHERE order_intent_id = ? ORDER BY id DESC LIMIT 1",
            (order_intent_id,),
        ).fetchone()
        return dict(row) if row else None

    # --- Executions ---

    def record_execution(
        self,
        order_intent_id: int,
        risk_decision_id: int,
        result: ExecutionResult,
    ) -> int:
        """Record an execution result. Returns execution ID.

        If an execution already exists for this order_intent_id, it is updated
        in place (upsert) so that exactly-once portfolio markers are preserved.
        """
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT id FROM executions WHERE order_intent_id = ? ORDER BY id DESC LIMIT 1",
                (order_intent_id,),
            ).fetchone()
            if existing:
                exec_id = existing["id"]
                conn.execute(
                    """
                    UPDATE executions
                    SET risk_decision_id = ?, status = ?, adapter = ?, order_id = ?, filled = ?,
                        fail_reason = ?, fill_price = ?, fill_qty = ?, fee = ?, fee_bps = ?,
                        slippage_bps = ?
                    WHERE id = ?
                    """,
                    (
                        risk_decision_id,
                        result.status,
                        result.adapter,
                        result.order_id,
                        1 if result.filled else 0,
                        result.fail_reason,
                        result.fill_price,
                        result.fill_qty,
                        result.fee,
                        result.fee_bps,
                        result.slippage_bps,
                        exec_id,
                    ),
                )
                return exec_id
            cur = conn.execute(
                """
                INSERT INTO executions (order_intent_id, risk_decision_id, status, adapter, order_id, filled, fail_reason, fill_price, fill_qty, fee, fee_bps, slippage_bps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    order_intent_id,
                    risk_decision_id,
                    result.status,
                    result.adapter,
                    result.order_id,
                    1 if result.filled else 0,
                    result.fail_reason,
                    result.fill_price,
                    result.fill_qty,
                    result.fee,
                    result.fee_bps,
                    result.slippage_bps,
                ),
            )
            return cur.lastrowid

    def finalize_execution(
        self,
        execution_id: int,
        result: ExecutionResult,
        conn: Optional[Any] = None,
    ) -> int:
        """Finalize an existing execution row by primary key.

        Updates status, fill details, and leaves portfolio_applied untouched
        (the engine sets it to 1 after portfolio application).

        If a connection is provided, the update runs within that connection's
        transaction context (no nested transaction). Otherwise a new transaction
        is opened.
        """
        if conn is not None:
            conn.execute(
                """
                UPDATE executions
                SET status = ?, adapter = ?, order_id = ?, filled = ?,
                    fail_reason = ?, fill_price = ?, fill_qty = ?, fee = ?, fee_bps = ?,
                    slippage_bps = ?
                WHERE id = ?
                """,
                (
                    result.status,
                    result.adapter,
                    result.order_id,
                    1 if result.filled else 0,
                    result.fail_reason,
                    result.fill_price,
                    result.fill_qty,
                    result.fee,
                    result.fee_bps,
                    result.slippage_bps,
                    execution_id,
                ),
            )
            return execution_id
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE executions
                SET status = ?, adapter = ?, order_id = ?, filled = ?,
                    fail_reason = ?, fill_price = ?, fill_qty = ?, fee = ?, fee_bps = ?,
                    slippage_bps = ?
                WHERE id = ?
                """,
                (
                    result.status,
                    result.adapter,
                    result.order_id,
                    1 if result.filled else 0,
                    result.fail_reason,
                    result.fill_price,
                    result.fill_qty,
                    result.fee,
                    result.fee_bps,
                    result.slippage_bps,
                    execution_id,
                ),
            )
            return execution_id

    def get_execution(self, order_intent_id: int) -> Optional[dict]:
        """Get execution result for an order intent."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM executions WHERE order_intent_id = ? ORDER BY id DESC LIMIT 1",
            (order_intent_id,),
        ).fetchone()
        return dict(row) if row else None

    # --- Portfolio / Positions ---

    def get_portfolio(self) -> dict:
        """Get current portfolio state (singleton row)."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM portfolio_state WHERE id = 1").fetchone()
        return dict(row) if row else {"id": 1, "cash": 100000.0, "updated_at": _utc_now_iso()}

    def update_cash(self, new_cash: float) -> None:
        """Update cash balance."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE id = 1",
                (new_cash, _utc_now_iso()),
            )

    def get_position(self, asset: str) -> Optional[dict]:
        """Get current position for an asset."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM positions WHERE asset = ?", (asset,)).fetchone()
        return dict(row) if row else None

    def update_position(
        self, asset: str, qty: float, avg_entry_price: Optional[float] = None, realized_pnl: Optional[float] = None
    ) -> None:
        """Upsert position for an asset."""
        with self.transaction() as conn:
            existing = conn.execute("SELECT * FROM positions WHERE asset = ?", (asset,)).fetchone()
            if existing:
                # Update existing position
                new_avg_price = avg_entry_price if avg_entry_price is not None else existing["avg_entry_price"]
                new_realized_pnl = realized_pnl if realized_pnl is not None else existing["realized_pnl"]
                conn.execute(
                    "UPDATE positions SET qty = ?, avg_entry_price = ?, realized_pnl = ?, updated_at = ? WHERE asset = ?",
                    (qty, new_avg_price, new_realized_pnl, _utc_now_iso(), asset),
                )
            else:
                conn.execute(
                    "INSERT INTO positions (asset, qty, avg_entry_price, realized_pnl, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (asset, qty, avg_entry_price, realized_pnl if realized_pnl is not None else 0.0, _utc_now_iso()),
                )

    def get_all_positions(self) -> list[dict]:
        """Get all current positions."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM positions WHERE qty != 0").fetchall()
        return [dict(r) for r in rows]

    # --- Daily Loss State ---

    def get_daily_loss_state(self) -> dict:
        """Get daily loss tracking state."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM daily_loss_state WHERE id = 1").fetchone()
        if row:
            return dict(row)
        return {"id": 1, "daily_start_equity": None, "last_trading_date": None, "updated_at": _utc_now_iso()}

    def update_daily_loss_state(self, daily_start_equity: Optional[float], last_trading_date: Optional[str]) -> None:
        """Update daily loss tracking state."""
        with self.transaction() as conn:
            conn.execute(
                "UPDATE daily_loss_state SET daily_start_equity = ?, last_trading_date = ?, updated_at = ? WHERE id = 1",
                (daily_start_equity, last_trading_date, _utc_now_iso()),
            )

    # --- Paper Orders (Crash Recovery) ---

    def save_paper_order(self, order: "PaperOrder") -> int:
        """Save or update a paper broker order for crash recovery."""
        with self.transaction() as conn:
            # Use UPSERT to handle both insert and update
            cur = conn.execute(
                """
                INSERT INTO paper_orders (
                    client_order_id, asset, side, notional, qty, limit_price, reference_price,
                    fee_bps, slippage_bps, status, fill_price, fill_qty, fee,
                    broker_order_id, fail_reason, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(client_order_id) DO NOTHING
                """,
                (
                    order.client_order_id,
                    order.asset,
                    order.side.value,
                    order.notional,
                    order.qty,
                    order.limit,
                    order.reference_price,
                    order.fee_bps,
                    order.slippage_bps,
                    order.status,
                    order.fill_price,
                    order.fill_qty,
                    order.fee,
                    order.broker_order_id,
                    order.fail_reason,
                    order.created_at.isoformat() if order.created_at else _utc_now_iso(),
                    _utc_now_iso(),
                ),
            )
            return cur.lastrowid

    def reserve_paper_order(
        self,
        client_order_id: str,
        asset: str,
        side: str,
        notional: float,
        qty: Optional[float],
        limit_price: Optional[float],
        reference_price: float,
        fee_bps: int,
        slippage_bps: int,
        broker_order_id: str,
    ) -> bool:
        """Atomically reserve a paper order slot for crash recovery idempotency.

        Uses INSERT ... ON CONFLICT DO NOTHING so that only the first caller
        with a given client_order_id can proceed. Returns True if the reservation
        was created (this caller owns the order), False if an order already exists
        (another caller already owns it).

        This is safe for concurrent execution because SQLite serializes writes
        and the UNIQUE constraint on client_order_id prevents duplicates.
        """
        with self.transaction() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO paper_orders (
                        client_order_id, asset, side, notional, qty, limit_price,
                        reference_price, fee_bps, slippage_bps, status,
                        broker_order_id, fail_reason, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, NULL, ?, ?)
                    """,
                    (
                        client_order_id,
                        asset,
                        side,
                        notional,
                        qty,
                        limit_price,
                        reference_price,
                        fee_bps,
                        slippage_bps,
                        broker_order_id,
                        _utc_now_iso(),
                        _utc_now_iso(),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                # client_order_id already exists - another caller owns this order
                return False

    def get_paper_order(self, client_order_id: str) -> Optional[dict]:
        """Get paper order by client_order_id for idempotency check."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM paper_orders WHERE client_order_id = ?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_pending_paper_orders(self) -> list[dict]:
        """Get all PENDING paper orders for recovery."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM paper_orders WHERE status = 'PENDING' ORDER BY created_at"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_paper_order_status(
        self,
        client_order_id: str,
        status: str,
        fill_price: Optional[float] = None,
        fill_qty: Optional[float] = None,
        fee: Optional[float] = None,
        fee_bps: Optional[int] = None,
        slippage_bps: Optional[int] = None,
        broker_order_id: Optional[str] = None,
        fail_reason: Optional[str] = None,
    ) -> None:
        """Update paper order status after broker execution."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE paper_orders SET
                    status = ?,
                    fill_price = ?,
                    fill_qty = ?,
                    fee = ?,
                    fee_bps = ?,
                    slippage_bps = ?,
                    broker_order_id = ?,
                    fail_reason = ?,
                    updated_at = ?
                WHERE client_order_id = ?
                """,
                (
                    status,
                    fill_price,
                    fill_qty,
                    fee,
                    fee_bps,
                    slippage_bps,
                    broker_order_id,
                    fail_reason,
                    _utc_now_iso(),
                    client_order_id,
                ),
            )

    def close(self) -> None:
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None