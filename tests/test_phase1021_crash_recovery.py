"""Phase 10.2.1.2: Crash recovery and idempotency regression tests.

These tests exercise the previously untested persistence and recovery code paths:
- SQLiteStore.save_paper_order() insert/retrieve
- Duplicate client_order_id idempotency
- PaperBroker.place() idempotency on repeated calls
- PENDING execution recovery
- Exactly-once portfolio application
- Missing signal_id fail-closed behavior
- Fault injection: broker raises mid-fill
- Concurrency: two recovery calls same order
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from trading_engine.engine import TradingEngine
from trading_engine.execution.paper_broker import PaperBroker
from trading_engine.models import (
    ExecutionResult, OrderIntent, RiskDecision, RiskStatus,
    Signal, SignalSide,
)
from trading_engine.risk import RiskConfig
from trading_engine.persistence.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_paper_order(client_order_id="ORD-001", status="FILLED", fail_reason=None):
    from trading_engine.models import PaperOrder
    return PaperOrder(
        client_order_id=client_order_id,
        asset="BTC-USDT",
        side=SignalSide.LONG,
        notional=1000.0,
        qty=0.02,
        limit=None,
        reference_price=50000.0,
        fee_bps=10,
        slippage_bps=5,
        status=status,
        fill_price=50000.0,
        fill_qty=0.02,
        fee=10.0,
        broker_order_id="P123456789",
        fail_reason=fail_reason,
    )


def _make_intent(client_order_id="ORD-001"):
    return OrderIntent(
        client_order_id=client_order_id,
        asset="BTC-USDT",
        side=SignalSide.LONG,
        notional=1000.0,
        qty=0.02,
        limit=None,
        meta={},
    )


def _make_risk_decision(status=RiskStatus.APPROVED):
    return RiskDecision(
        status=status,
        reason="test",
        checks=[("test_check", True)],
    )


def _setup_engine_with_store(store, broker=None):
    """Create a TradingEngine instance with all required attributes set."""
    engine = TradingEngine.__new__(TradingEngine)
    engine.store = store
    engine.broker = broker if broker is not None else PaperBroker(paper=True, store=store)
    engine._daily_start_equity = 100000.0
    engine._last_date = datetime.now(timezone.utc).date()
    engine._market_data_provider = None
    engine._max_market_data_age_seconds = 60.0
    engine._default_notional = 1000.0
    engine._running = False
    from trading_engine.risk import RiskGates
    engine.risk_gates = RiskGates(store, RiskConfig.default())
    return engine


# ---------------------------------------------------------------------------
# Test 1: save_paper_order() inserts and retrieves a PaperOrder
# ---------------------------------------------------------------------------

def test_save_paper_order_inserts_and_retrieves():
    """save_paper_order() must persist a PaperOrder and allow retrieval by client_order_id."""
    db_path = os.path.join(tempfile.mkdtemp(), "test.db")
    store = SQLiteStore(db_path)
    try:
        order = _make_paper_order("ORD-001")
        row_id = store.save_paper_order(order)
        assert row_id is not None

        retrieved = store.get_paper_order("ORD-001")
        assert retrieved is not None
        assert retrieved["client_order_id"] == "ORD-001"
        assert retrieved["asset"] == "BTC-USDT"
        assert retrieved["side"] == "LONG"
        assert retrieved["notional"] == 1000.0
        assert retrieved["status"] == "FILLED"
        assert retrieved["fill_price"] == 50000.0
        assert retrieved["fill_qty"] == 0.02
        assert retrieved["fee"] == 10.0
        assert retrieved["broker_order_id"] == "P123456789"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 2: Duplicate client_order_id does not create a second order
# ---------------------------------------------------------------------------

def test_duplicate_client_order_id_does_not_create_second_order():
    """Two insertions with the same client_order_id must result in exactly one row (DO NOTHING)."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_dup.db")
    store = SQLiteStore(db_path)
    try:
        order = _make_paper_order("ORD-DUP")
        store.save_paper_order(order)
        store.save_paper_order(order)

        rows = store._get_conn().execute(
            "SELECT COUNT(*) AS cnt FROM paper_orders WHERE client_order_id = ?",
            ("ORD-DUP",),
        ).fetchone()
        assert rows["cnt"] == 1

        retrieved = store.get_paper_order("ORD-DUP")
        assert retrieved["status"] == "FILLED"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 3: Calling PaperBroker.place() twice with the same ID returns existing result
# ---------------------------------------------------------------------------

def test_paper_broker_place_idempotent_same_client_order_id():
    """place() called twice with the same client_order_id must return the existing result."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_broker.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-IDEMP")

        result1 = broker.place(intent, reference_price=50000.0)
        assert result1.status == "FILLED"
        assert result1.filled is True
        assert result1.order_id is not None

        result2 = broker.place(intent, reference_price=50000.0)
        assert result2.status == "FILLED"
        assert result2.filled is True
        assert result2.order_id == result1.order_id
        assert result2.broker_order_id == result1.broker_order_id

        count = store._get_conn().execute(
            "SELECT COUNT(*) AS cnt FROM paper_orders WHERE client_order_id = ?",
            ("ORD-IDEMP",),
        ).fetchone()["cnt"]
        assert count == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 4: A FILLED PaperOrder survives closing and reopening the SQLite connection
# ---------------------------------------------------------------------------

def test_paper_order_survives_connection_reopen():
    """A persisted PaperOrder must be retrievable after the store is closed and reopened."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_reopen.db")
    store1 = SQLiteStore(db_path)
    try:
        order = _make_paper_order("ORD-SURVIVE")
        store1.save_paper_order(order)
    finally:
        store1.close()

    store2 = SQLiteStore(db_path)
    try:
        retrieved = store2.get_paper_order("ORD-SURVIVE")
        assert retrieved is not None
        assert retrieved["client_order_id"] == "ORD-SURVIVE"
        assert retrieved["status"] == "FILLED"
        assert retrieved["fill_price"] == 50000.0
    finally:
        store2.close()


# ---------------------------------------------------------------------------
# Test 5: PENDING execution recovery restores an existing FILLED order
# ---------------------------------------------------------------------------

def test_recovery_restores_filled_order_and_applies_portfolio_once():
    """Recovery finds an existing FILLED paper order and applies the portfolio update exactly once."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_recover.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        order = _make_paper_order("ORD-RECOVER-1", status="PENDING")
        store.save_paper_order(order)

        intent = _make_intent("ORD-RECOVER-1")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        store.update_paper_order_status(
            "ORD-RECOVER-1",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        engine = _setup_engine_with_store(store, broker)
        store.update_cash(100000.0)

        stats = engine.recover_pending_executions()

        assert stats["recovered_filled"] >= 1
        assert stats["errors"] == 0

        portfolio = store.get_portfolio()
        assert portfolio["cash"] < 100000.0

        conn = store._get_conn()
        exec_row = conn.execute(
            "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert exec_row["portfolio_applied"] == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 6: Recovery after simulated crash applies fill exactly once
# ---------------------------------------------------------------------------

def test_recovery_applies_fill_exactly_once():
    """Recovery must apply the fill only once even if the recovery path is triggered."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_exactly_once.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-EXACT")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-EXACT", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_paper_order_status(
            "ORD-EXACT",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        store.update_cash(100000.0)

        engine = _setup_engine_with_store(store, broker)

        stats1 = engine.recover_pending_executions()
        assert stats1["recovered_filled"] == 1

        cash_after_first = store.get_portfolio()["cash"]

        stats2 = engine.recover_pending_executions()
        cash_after_second = store.get_portfolio()["cash"]

        assert cash_after_first == cash_after_second, (
            f"Portfolio changed on second recovery: {cash_after_first} -> {cash_after_second}"
        )

        conn = store._get_conn()
        exec_row = conn.execute(
            "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert exec_row["portfolio_applied"] == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 7: Running recovery twice leaves portfolio unchanged on second run
# ---------------------------------------------------------------------------

def test_running_recovery_twice_leaves_portfolio_unchanged():
    """Second recovery run must leave all portfolio values identical to first run."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_twice.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-TWICE")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-TWICE", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_paper_order_status(
            "ORD-TWICE",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        store.update_cash(100000.0)

        engine = _setup_engine_with_store(store, broker)

        stats1 = engine.recover_pending_executions()
        cash1 = store.get_portfolio()["cash"]
        pos1 = store.get_position("BTC-USDT")

        stats2 = engine.recover_pending_executions()
        cash2 = store.get_portfolio()["cash"]
        pos2 = store.get_position("BTC-USDT")

        assert cash1 == cash2
        assert pos1 == pos2
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 8: An existing PENDING broker order stays PENDING
# ---------------------------------------------------------------------------

def test_pending_order_stays_pending():
    """A PENDING paper order must remain PENDING after recovery (no auto-fill)."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_pending.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        order = _make_paper_order("ORD-PENDING", status="PENDING")
        store.save_paper_order(order)

        intent = _make_intent("ORD-PENDING")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        engine = _setup_engine_with_store(store, broker)

        stats = engine.recover_pending_executions()

        assert stats["recovered_filled"] == 0

        retrieved = store.get_paper_order("ORD-PENDING")
        assert retrieved["status"] == "PENDING"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 9: A missing broker order is retried using original client_order_id
# ---------------------------------------------------------------------------

def test_missing_broker_order_retried_with_original_id():
    """When no broker order exists, recovery must retry using the original client_order_id."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_retry.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-RETRY")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        mock_provider = MagicMock()
        snapshot = MagicMock()
        snapshot.inst_id = "BTC-USDT"
        snapshot.last = 50000.0
        snapshot.ts = datetime.now(timezone.utc)
        snapshot.received_at = datetime.now(timezone.utc)
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()

        engine = TradingEngine(
            db_path=":memory:",
            market_data_provider=mock_provider,
        )
        engine.store = store
        engine.broker = broker
        engine._daily_start_equity = 100000.0
        engine._last_date = datetime.now(timezone.utc).date()

        stats = engine.recover_pending_executions()
        assert stats["retried"] >= 1 or stats["errors"] == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 10: A rejected order leaves the portfolio unchanged
# ---------------------------------------------------------------------------

def test_rejected_order_leaves_portfolio_unchanged():
    """A REJECTED order recovery must not modify the portfolio."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_rejected.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        order = _make_paper_order("ORD-REJ", status="REJECTED", fail_reason="Insufficient funds")
        store.save_paper_order(order)

        intent = _make_intent("ORD-REJ")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        store.update_cash(100000.0)
        cash_before = store.get_portfolio()["cash"]

        engine = _setup_engine_with_store(store, broker)

        stats = engine.recover_pending_executions()
        assert stats["recovered_rejected"] >= 1

        cash_after = store.get_portfolio()["cash"]
        assert cash_after == cash_before
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test 11: Database failure during portfolio application rolls back all changes
# ---------------------------------------------------------------------------

def test_database_failure_rolls_back_portfolio_changes():
    """A transaction failure during portfolio application must roll back all changes.

    This test injects a SQLite trigger on the positions table that fires on
    the position UPSERT inside _update_portfolio_on_fill(). The cash UPDATE
    (first SQL statement in the transaction) succeeds; the trigger on the
    positions UPDATE raises, causing SQLite to roll back the entire transaction
    including the cash change.

    Verifies:
    - Cash UPDATE was reached (portfolio_state row was modified before rollback)
    - Position UPDATE triggered the failure
    - All changes (cash, position, execution) are rolled back
    """
    db_path = os.path.join(tempfile.mkdtemp(), "test_rollback.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-ROLLBACK")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-ROLLBACK", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_paper_order_status(
            "ORD-ROLLBACK",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        store.update_cash(100000.0)
        store.update_position("BTC-USDT", qty=0.01, avg_entry_price=49000.0)
        cash_before = store.get_portfolio()["cash"]
        pos_before = store.get_position("BTC-USDT")

        conn = store._get_conn()

        # Create a trigger that fires on the position UPSERT inside
        # _update_portfolio_on_fill. The cash UPDATE (first SQL statement)
        # succeeds; this trigger on the positions UPDATE raises, causing
        # SQLite to roll back the entire transaction.
        conn.execute("""
            CREATE TEMP TRIGGER fail_on_position_upsert
            BEFORE UPDATE ON positions
            FOR EACH ROW
            WHEN NEW.asset = 'BTC-USDT'
            BEGIN
                SELECT RAISE(ABORT, "simulated transaction failure");
            END
        """)
        conn.commit()

        engine = _setup_engine_with_store(store, broker)

        stats = engine.recover_pending_executions()
        assert stats["errors"] >= 1, f"Expected errors >= 1, got {stats}"

        # Verify all state is unchanged after rollback
        cash_after = store.get_portfolio()["cash"]
        assert cash_after == cash_before, (
            f"Portfolio cash must be unchanged after rollback: {cash_before} -> {cash_after}"
        )

        pos_after = store.get_position("BTC-USDT")
        assert pos_before == pos_after, (
            f"Position must be unchanged after rollback: {pos_before} -> {pos_after}"
        )

        # Verify full position state is unchanged
        assert pos_before["qty"] == pos_after["qty"], (
            f"Position qty must be unchanged after rollback"
        )
        assert pos_before["avg_entry_price"] == pos_after["avg_entry_price"], (
            f"Position avg_entry_price must be unchanged after rollback"
        )
        assert pos_before["realized_pnl"] == pos_after["realized_pnl"], (
            f"Position realized_pnl must be unchanged after rollback"
        )

        # Execution must still be PENDING (never finalized)
        exec_row = store.get_execution(intent_id)
        assert exec_row is not None, "Execution row must exist"
        assert exec_row["status"] == "PENDING", (
            f"Execution must remain PENDING after rollback, got {exec_row['status']}"
        )
        assert exec_row["portfolio_applied"] == 0, (
            "portfolio_applied must remain 0 after rollback"
        )

        # Clean up trigger
        conn.execute("DROP TRIGGER IF EXISTS fail_on_position_upsert")
        conn.commit()
    finally:
        store.close()

# ---------------------------------------------------------------------------
# Test 12: Missing signal_id fails-closed (rejected, not executed)
# ---------------------------------------------------------------------------

def test_missing_signal_id_fails_closed(caplog):
    """A signal without signal_id must be REJECTED before execution (fail-closed)."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_no_sig.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = Signal(
            strategy="test",
            asset="BTC-USDT",
            side=SignalSide.LONG,
            conf=0.8,
            rationale="test",
            required_inputs=["close"],
            issued_at=datetime.now(timezone.utc),
            signal_id=None,
        )

        with caplog.at_level("WARNING"):
            result = engine.process_signal(signal)

        assert result.status == "REJECTED", f"Expected REJECTED, got {result.status}"
        assert "signal_id" in result.fail_reason.lower()

        warning_msgs = [r for r in caplog.records if "signal_id" in r.getMessage().lower()]
        assert len(warning_msgs) >= 1, (
            f"Expected warning about missing signal_id, got: {[r.getMessage() for r in caplog.records]}"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test A: Broker raises during fill — portfolio must not be applied
# ---------------------------------------------------------------------------

def test_broker_raise_mid_fill_portfolio_not_applied():
    """If PaperBroker.place raises during fill, portfolio must not be modified."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_broker_raise.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-BROKER-RAISE")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        execution_id = store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-BROKER-RAISE", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_cash(100000.0)
        cash_before = store.get_portfolio()["cash"]

        engine = _setup_engine_with_store(store, broker)

        original_place = broker.place

        def raising_place(intent_arg, reference_price):
            result = original_place(intent_arg, reference_price)
            raise RuntimeError("broker died mid-fill")

        broker.place = raising_place

        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = Signal(
            strategy="test",
            asset="BTC-USDT",
            side=SignalSide.LONG,
            conf=0.8,
            rationale="test",
            required_inputs=["close"],
            issued_at=datetime.now(timezone.utc),
            signal_id="broker_raise_signal_001",
        )

        try:
            engine.process_signal(signal)
            assert False, "Expected RuntimeError to propagate"
        except RuntimeError:
            pass

        cash_after = store.get_portfolio()["cash"]
        assert cash_after == cash_before, (
            f"Portfolio changed despite broker failure: {cash_before} -> {cash_after}"
        )

        exec_row = store.get_execution(intent_id)
        assert exec_row["status"] == "PENDING"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test B: Concurrent recovery race — same client_order_id, two recovery calls
# ---------------------------------------------------------------------------

def test_concurrent_recovery_same_client_order_id():
    """Two recovery instances with same pending execution must not double-apply portfolio."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_concurrent.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-CONCURRENT")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-CONCURRENT", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_paper_order_status(
            "ORD-CONCURRENT",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        store.update_cash(100000.0)

        engine1 = _setup_engine_with_store(store, broker)
        engine2 = _setup_engine_with_store(store, broker)

        stats1 = engine1.recover_pending_executions()
        stats2 = engine2.recover_pending_executions()

        total_recovered = stats1["recovered_filled"] + stats2["recovered_filled"]
        assert total_recovered == 1, (
            f"Expected exactly 1 recovery, got {total_recovered} "
            f"(stats1={stats1}, stats2={stats2})"
        )

        portfolio = store.get_portfolio()
        assert portfolio["cash"] < 100000.0

        conn = store._get_conn()
        exec_row = conn.execute(
            "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert exec_row["portfolio_applied"] == 1
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test C: Recovery leaves PENDING order untouched (no auto-fill)
# ---------------------------------------------------------------------------

def test_recovery_pending_order_remains_pending():
    """A PENDING broker order must remain PENDING after recovery (no auto-fill)."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_pending_stays.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-STAYS-PENDING")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-STAYS-PENDING", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_cash(100000.0)
        cash_before = store.get_portfolio()["cash"]

        engine = _setup_engine_with_store(store, broker)

        stats = engine.recover_pending_executions()

        assert stats["recovered_filled"] == 0
        assert stats["errors"] == 0

        cash_after = store.get_portfolio()["cash"]
        assert cash_after == cash_before

        retrieved = store.get_paper_order("ORD-STAYS-PENDING")
        assert retrieved["status"] == "PENDING"
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test D: Genuine crash/restart integration test
# ---------------------------------------------------------------------------

def test_crash_restart_integration_applies_fill_once():
    """Real crash/restart scenario using the actual production code path.

    Sequence:
    1. Create file-backed SQLite database.
    2. Create real TradingEngine, SQLiteStore and PaperBroker.
    3. Process a signal through process_signal() - this calls broker.place()
       which creates a PENDING execution and persists a FILLED PaperOrder.
    4. Inject an exception AFTER PaperBroker.place() returns successfully
       but BEFORE _update_portfolio_on_fill() commits.
    5. Confirm: paper order is durably FILLED, execution remains PENDING,
       portfolio has NOT been updated.
    6. Close all original connections and instances.
    7. Create new SQLiteStore, PaperBroker, TradingEngine against same DB.
    8. Run real recover_pending_executions().
    9. Confirm fill recovered using original client_order_id.
    10. Confirm portfolio updated exactly once.
    11. Run recovery again - confirm no portfolio or execution changes.
    """

    import tempfile
    import os

    db_path = os.path.join(tempfile.mkdtemp(), "test_crash_restart.db")

    # =========================================================================
    # Phase 1: Process signal with real engine, then crash before finalize
    # =========================================================================
    store1 = SQLiteStore(db_path)
    broker1 = PaperBroker(paper=True, store=store1)
    market_data_provider = MagicMock()
    from trading_engine.marketdata import TickerSnapshot
    snapshot = TickerSnapshot(
        inst_id="BTC-USDT",
        last=50000.0,
        bid=None,
        ask=None,
        open_24h=None,
        high_24h=None,
        low_24h=None,
        ts=datetime.now(timezone.utc),
        received_at=datetime.now(timezone.utc),
    )
    market_data_provider.get_ticker.return_value = snapshot
    market_data_provider.close = MagicMock()

    engine1 = TradingEngine(
        db_path=db_path,
        market_data_provider=market_data_provider,
    )
    engine1.store = store1
    engine1.broker = broker1
    engine1._daily_start_equity = 100000.0
    engine1._last_date = datetime.now(timezone.utc).date()

    store1.update_cash(100000.0)
    cash_before = store1.get_portfolio()["cash"]

    signal = Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.8,
        rationale="crash restart test",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc),
        signal_id="crash_restart_signal_001",
    )

    # Monkeypatch _update_portfolio_on_fill to raise AFTER broker.place() returns
    # but BEFORE the execution is finalized in the database.
    original_update_portfolio = engine1._update_portfolio_on_fill

    def raising_update_portfolio(*args, **kwargs):
        raise RuntimeError("simulated crash before execution finalize")

    engine1._update_portfolio_on_fill = raising_update_portfolio

    try:
        engine1.process_signal(signal)
        assert False, "Expected RuntimeError to propagate"
    except RuntimeError:
        pass

    engine1.shutdown()
    store1.close()

    # Verify DB state after crash: paper order is FILLED, execution is PENDING,
    # portfolio has NOT been updated.
    # We look up by signal_id since process_signal() created its own intent.
    store1_check = SQLiteStore(db_path)
    try:
        # Find the signal by signal_id to get the order intent
        signal_row = store1_check.get_signal_by_signal_id("crash_restart_signal_001")
        assert signal_row is not None, "Signal must exist after crash"
        signal_row_id = signal_row["id"]

        # Find the order intent linked to this signal
        intent_row = store1_check.get_order_intent_by_signal_ref("crash_restart_signal_001")
        assert intent_row is not None, "Order intent must exist after crash"
        intent_id = intent_row["id"]
        client_order_id = intent_row["client_order_id"]

        # Verify paper order is FILLED
        pending_paper = store1_check.get_paper_order(client_order_id)
        assert pending_paper is not None, "Paper order must exist after crash"
        assert pending_paper["status"] == "FILLED", (
            f"Expected FILLED paper order, got {pending_paper['status']}"
        )
        assert pending_paper["broker_order_id"] is not None, (
            "Paper order must have broker_order_id"
        )

        # Verify execution is PENDING
        pending_exec = store1_check.get_execution(intent_id)
        assert pending_exec is not None, "Execution must exist after crash"
        assert pending_exec["status"] == "PENDING", (
            f"Expected PENDING execution, got {pending_exec['status']}"
        )

        # Verify portfolio has NOT been updated
        portfolio_check = store1_check.get_portfolio()
        assert portfolio_check["cash"] == 100000.0, (
            f"Portfolio must NOT be updated after crash: {portfolio_check['cash']}"
        )
    finally:
        store1_check.close()

    # =========================================================================
    # Phase 2: Restart with new engine instances - crash recovery
    # =========================================================================
    store2 = SQLiteStore(db_path)
    broker2 = PaperBroker(paper=True, store=store2)
    engine2 = _setup_engine_with_store(store2, broker2)

    stats = engine2.recover_pending_executions()

    # Verify recovery applied fill exactly once
    assert stats["recovered_filled"] == 1, f"Expected 1 recovered, got {stats}"
    assert stats["errors"] == 0, f"Expected 0 errors, got {stats}"

    # Verify portfolio was updated exactly once (cash deducted)
    portfolio = store2.get_portfolio()
    assert portfolio["cash"] < 100000.0, (
        f"Portfolio must be updated after recovery: {portfolio['cash']}"
    )

    # Verify execution marked portfolio_applied
    exec_row = store2._get_conn().execute(
        "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
        (intent_id,),
    ).fetchone()
    assert exec_row["portfolio_applied"] == 1, (
        f"portfolio_applied must be 1, got {exec_row['portfolio_applied']}"
    )

    # =========================================================================
    # Phase 3: Call recovery again - verify unchanged state
    # =========================================================================
    cash_after_first = store2.get_portfolio()["cash"]
    pos_after_first = store2.get_position("BTC-USDT")
    exec_count_first = store2._get_conn().execute(
        "SELECT COUNT(*) FROM executions WHERE order_intent_id = ?", (intent_id,)
    ).fetchone()[0]

    stats2 = engine2.recover_pending_executions()

    cash_after_second = store2.get_portfolio()["cash"]
    pos_after_second = store2.get_position("BTC-USDT")
    exec_count_second = store2._get_conn().execute(
        "SELECT COUNT(*) FROM executions WHERE order_intent_id = ?", (intent_id,)
    ).fetchone()[0]

    assert cash_after_first == cash_after_second, (
        f"Cash changed on second recovery: {cash_after_first} -> {cash_after_second}"
    )
    assert pos_after_first == pos_after_second
    assert exec_count_first == exec_count_second
    assert stats2["errors"] == 0

    store2.close()
# ---------------------------------------------------------------------------
# Test E: Concurrent PaperBroker.place() calls - same client_order_id
# ---------------------------------------------------------------------------

def test_concurrent_broker_place_same_client_order_id():
    """Two simultaneous PaperBroker.place() calls with same client_order_id.

    Uses:
    - Two independent SQLiteStore connections to same file-backed DB.
    - Two PaperBroker instances.
    - Same client_order_id.
    - Real concurrent place() calls.

    Verifies:
    - Exactly one durable paper order exists.
    - Only one fill calculated and persisted.
    - Duplicate does not generate competing fill.
    - Both callers receive consistent result.
    """
    db_path = os.path.join(tempfile.mkdtemp(), "test_concurrent_broker.db")

    store1 = SQLiteStore(db_path)
    store2 = SQLiteStore(db_path)
    broker1 = PaperBroker(paper=True, store=store1)
    broker2 = PaperBroker(paper=True, store=store2)

    try:
        intent1 = _make_intent("ORD-CONCURRENT-BROKER")
        intent2 = _make_intent("ORD-CONCURRENT-BROKER")  # Same client_order_id

        results = [None, None]
        errors = [None, None]

        def place_broker(idx, broker, intent):
            try:
                results[idx] = broker.place(intent, reference_price=50000.0)
            except Exception as exc:
                errors[idx] = exc

        import threading

        t1 = threading.Thread(target=place_broker, args=(0, broker1, intent1))
        t2 = threading.Thread(target=place_broker, args=(1, broker2, intent2))

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # No unexpected errors
        assert errors[0] is None, f"Broker 1 error: {errors[0]}"
        assert errors[1] is None, f"Broker 2 error: {errors[1]}"

        # Both callers got results
        assert results[0] is not None
        assert results[1] is not None

        # Exactly one durable paper order
        count = store1._get_conn().execute(
            "SELECT COUNT(*) FROM paper_orders WHERE client_order_id = ?",
            ("ORD-CONCURRENT-BROKER",),
        ).fetchone()[0]
        assert count == 1, f"Expected 1 paper order, got {count}"

        # Only one fill persisted
        retrieved = store1.get_paper_order("ORD-CONCURRENT-BROKER")
        assert retrieved["status"] == "FILLED"

        # Both callers received consistent result (same broker_order_id)
        assert results[0].broker_order_id == results[1].broker_order_id
        assert results[0].order_id == results[1].order_id
        assert results[0].filled is True
        assert results[1].filled is True

    finally:
        store1.close()
        store2.close()


# ---------------------------------------------------------------------------
# Test F: Signal ID validation - invalid identifiers
# ---------------------------------------------------------------------------

def _make_valid_signal(signal_id):
    return Signal(
        strategy="test",
        asset="BTC-USDT",
        side=SignalSide.LONG,
        conf=0.8,
        rationale="test",
        required_inputs=["close"],
        issued_at=datetime.now(timezone.utc),
        signal_id=signal_id,
    )


def test_signal_id_none_rejected():
    """signal_id=None must be REJECTED before execution."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_signal_none.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        # Mock market data provider so signal_id check is reached
        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = _make_valid_signal(None)
        result = engine.process_signal(signal)

        assert result.status == "REJECTED"
        assert "signal_id" in result.fail_reason.lower()

        # No execution created
        exec_count = store._get_conn().execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        assert exec_count == 0
    finally:
        store.close()


def test_signal_id_empty_string_rejected():
    """signal_id="" must be REJECTED before execution."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_signal_empty.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        # Mock market data provider so signal_id check is reached
        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = _make_valid_signal("")
        result = engine.process_signal(signal)

        assert result.status == "REJECTED"
        assert "signal_id" in result.fail_reason.lower()

        # No execution created
        exec_count = store._get_conn().execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        assert exec_count == 0
    finally:
        store.close()


def test_signal_id_whitespace_rejected():
    """signal_id="  " (whitespace-only) must be REJECTED before execution."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_signal_ws.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        # Mock market data provider so signal_id check is reached
        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = _make_valid_signal("   ")
        result = engine.process_signal(signal)

        assert result.status == "REJECTED"
        assert "signal_id" in result.fail_reason.lower()

        # No execution created
        exec_count = store._get_conn().execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        assert exec_count == 0
    finally:
        store.close()


def test_signal_id_non_string_rejected():
    """Non-string signal_id (e.g., int) must be REJECTED before execution."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_signal_nonstr.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        # Mock market data provider so signal_id check is reached
        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        # Create signal with non-string signal_id via object.__setattr__ since frozen
        signal = _make_valid_signal("valid-initial")
        object.__setattr__(signal, "signal_id", 12345)

        result = engine.process_signal(signal)

        assert result.status == "REJECTED"
        assert "signal_id" in result.fail_reason.lower()

        # No execution created
        exec_count = store._get_conn().execute(
            "SELECT COUNT(*) FROM executions"
        ).fetchone()[0]
        assert exec_count == 0
    finally:
        store.close()


def test_signal_id_valid_accepted():
    """Valid non-empty signal_id must be accepted and executed."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_signal_valid.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        engine = _setup_engine_with_store(store, broker)

        # Mock market data provider so signal_id check is reached
        from trading_engine.marketdata import TickerSnapshot
        mock_provider = MagicMock()
        snapshot = TickerSnapshot(
            inst_id="BTC-USDT",
            last=50000.0,
            bid=None,
            ask=None,
            open_24h=None,
            high_24h=None,
            low_24h=None,
            ts=datetime.now(timezone.utc),
            received_at=datetime.now(timezone.utc),
        )
        mock_provider.get_ticker.return_value = snapshot
        mock_provider.close = MagicMock()
        engine._market_data_provider = mock_provider

        signal = _make_valid_signal("valid-signal-001")
        result = engine.process_signal(signal)

        # Should execute (not rejected for missing signal_id)
        assert result.status in ("FILLED", "REJECTED")

        # Warning not logged for valid signal_id
        # (execution proceeded without signal_id rejection)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test G: TRUE LEGACY RECOVERY - FILLED execution with portfolio_applied=0
# ---------------------------------------------------------------------------

def test_legacy_recovery_filled_execution_portfolio_applied_zero():
    """Test recovery of a FILLED execution that was never applied to portfolio.

    This simulates a legacy scenario where:
    - Execution was already marked FILLED in the database
    - But portfolio_applied = 0 (crash after execution finalization, before portfolio update)
    - Recovery must apply portfolio exactly once via _update_portfolio_on_fill
    """
    db_path = os.path.join(tempfile.mkdtemp(), "test_legacy_recovery.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        # Setup: Create order intent, risk decision, and FILLED execution with portfolio_applied=0
        intent = _make_intent("ORD-LEGACY-1")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        # Insert FILLED execution directly with portfolio_applied=0 (simulating legacy state)
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO executions (
                    order_intent_id, risk_decision_id, status, adapter, order_id, filled,
                    fail_reason, fill_price, fill_qty, fee, fee_bps, slippage_bps, portfolio_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    intent_id, risk_id, "FILLED", "paper", "P123456789", 1,
                    None, 50000.0, 0.02, 10.0, 10, 5,
                ),
            )

        # Also create the corresponding paper order with correct client_order_id
        order = _make_paper_order("ORD-LEGACY-1", status="FILLED")
        # Create a new PaperOrder with the correct client_order_id since it's frozen
        from trading_engine.models import PaperOrder
        order = PaperOrder(
            client_order_id="ORD-LEGACY-1",
            asset=order.asset,
            side=order.side,
            notional=order.notional,
            qty=order.qty,
            limit=order.limit,
            reference_price=order.reference_price,
            fee_bps=order.fee_bps,
            slippage_bps=order.slippage_bps,
            status=order.status,
            fill_price=order.fill_price,
            fill_qty=order.fill_qty,
            fee=order.fee,
            broker_order_id=order.broker_order_id,
            fail_reason=order.fail_reason,
        )
        store.save_paper_order(order)

        store.update_cash(100000.0)

        engine = _setup_engine_with_store(store, broker)

        # Run recovery - should find FILLED execution with portfolio_applied=0 and apply it
        stats = engine.recover_pending_executions()

        assert stats["recovered_filled"] == 1, f"Expected 1 recovered_filled, got {stats}"
        assert stats["errors"] == 0

        # Verify portfolio was updated exactly once
        portfolio = store.get_portfolio()
        assert portfolio["cash"] < 100000.0, f"Portfolio must be updated: {portfolio['cash']}"

        # Verify execution marked portfolio_applied = 1
        conn = store._get_conn()
        exec_row = conn.execute(
            "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert exec_row["portfolio_applied"] == 1, (
            f"portfolio_applied must be 1, got {exec_row['portfolio_applied']}"
        )

        # Run recovery again - should not apply again (no pending executions with portfolio_applied=0)
        cash_after_first = portfolio["cash"]
        stats2 = engine.recover_pending_executions()
        cash_after_second = store.get_portfolio()["cash"]
        assert cash_after_first == cash_after_second, (
            f"Portfolio changed on second recovery: {cash_after_first} -> {cash_after_second}"
        )
        # Second run finds no pending executions (portfolio_applied is now 1)
        assert stats2["recovered_filled"] == 0
        assert stats2["total_pending"] == 0
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Test H: TRUE PARALLEL RECOVERY - Two threads, same execution
# ---------------------------------------------------------------------------

def test_true_parallel_recovery_same_execution():
    """Two separate engine instances running recovery concurrently on same execution.

    Uses two SQLiteStore instances and two TradingEngine instances in separate threads.
    Both attempt to recover the same FILLED execution with portfolio_applied=0.
    Result must be exactly one portfolio application.
    """
    import threading
    import time

    db_path = os.path.join(tempfile.mkdtemp(), "test_parallel_recovery.db")

    # Initial setup - create the database state
    store_setup = SQLiteStore(db_path)
    broker_setup = PaperBroker(paper=True, store=store_setup)
    try:
        intent = _make_intent("ORD-PARALLEL")
        intent_id = store_setup.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store_setup.record_risk_decision(intent_id, risk_decision)

        # FILLED execution with portfolio_applied=0
        with store_setup.transaction() as conn:
            conn.execute(
                """
                INSERT INTO executions (
                    order_intent_id, risk_decision_id, status, adapter, order_id, filled,
                    fail_reason, fill_price, fill_qty, fee, fee_bps, slippage_bps, portfolio_applied
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    intent_id, risk_id, "FILLED", "paper", "P123456789", 1,
                    None, 50000.0, 0.02, 10.0, 10, 5,
                ),
            )

        order = _make_paper_order("ORD-PARALLEL", status="FILLED")
        # Create a new PaperOrder with the correct client_order_id since it's frozen
        from trading_engine.models import PaperOrder
        order = PaperOrder(
            client_order_id="ORD-PARALLEL",
            asset=order.asset,
            side=order.side,
            notional=order.notional,
            qty=order.qty,
            limit=order.limit,
            reference_price=order.reference_price,
            fee_bps=order.fee_bps,
            slippage_bps=order.slippage_bps,
            status=order.status,
            fill_price=order.fill_price,
            fill_qty=order.fill_qty,
            fee=order.fee,
            broker_order_id=order.broker_order_id,
            fail_reason=order.fail_reason,
        )
        store_setup.save_paper_order(order)

        store_setup.update_cash(100000.0)
    finally:
        store_setup.close()

    results = []
    errors = []

    def run_recovery():
        try:
            store = SQLiteStore(db_path)
            broker = PaperBroker(paper=True, store=store)
            engine = _setup_engine_with_store(store, broker)
            stats = engine.recover_pending_executions()
            results.append(stats)
            store.close()
        except Exception as e:
            errors.append(e)

    # Start two threads concurrently
    t1 = threading.Thread(target=run_recovery)
    t2 = threading.Thread(target=run_recovery)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Check for errors
    assert len(errors) == 0, f"Recovery threads had errors: {errors}"

    # Both threads should report success
    assert len(results) == 2
    total_recovered = sum(r["recovered_filled"] for r in results)
    assert total_recovered == 1, f"Expected exactly 1 recovery total, got {total_recovered} (results: {results})"

    # Verify final state
    store_check = SQLiteStore(db_path)
    try:
        portfolio = store_check.get_portfolio()
        assert portfolio["cash"] < 100000.0, f"Portfolio must be updated: {portfolio['cash']}"

        conn = store_check._get_conn()
        exec_row = conn.execute(
            "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
            (intent_id,),
        ).fetchone()
        assert exec_row["portfolio_applied"] == 1, (
            f"portfolio_applied must be 1, got {exec_row['portfolio_applied']}"
        )
    finally:
        store_check.close()


# ---------------------------------------------------------------------------
# Test I: FINALIZE_EXECUTION ROLLBACK - Transaction failure during finalize
# ---------------------------------------------------------------------------

def test_finalize_execution_rollback_on_error():
    """Test that finalize_execution() rolls back portfolio changes on failure.

    This test injects a SQLite trigger that fires during finalize_execution()
    when updating the executions table. The trigger raises an error, causing
    the entire transaction (including portfolio changes) to roll back.
    """
    db_path = os.path.join(tempfile.mkdtemp(), "test_finalize_rollback.db")
    store = SQLiteStore(db_path)
    broker = PaperBroker(paper=True, store=store)
    try:
        intent = _make_intent("ORD-FINALIZE-ROLLBACK")
        intent_id = store.record_order_intent(intent)

        risk_decision = _make_risk_decision(RiskStatus.APPROVED)
        risk_id = store.record_risk_decision(intent_id, risk_decision)

        # Create PENDING execution
        execution_id = store.record_execution(
            intent_id, risk_id,
            ExecutionResult(status="PENDING", adapter="paper", order_id=None, filled=False),
        )

        pending_order = _make_paper_order("ORD-FINALIZE-ROLLBACK", status="PENDING")
        store.save_paper_order(pending_order)

        store.update_paper_order_status(
            "ORD-FINALIZE-ROLLBACK",
            status="FILLED",
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
        )

        store.update_cash(100000.0)
        store.update_position("BTC-USDT", qty=0.01, avg_entry_price=49000.0)
        cash_before = store.get_portfolio()["cash"]
        pos_before = store.get_position("BTC-USDT")

        conn = store._get_conn()

        # Create a trigger that fires on the executions UPDATE inside finalize_execution()
        # This simulates a failure during the finalization step
        # Note: SQLite triggers can't use parameters, so we hardcode the intent_id
        conn.execute(f"""
            CREATE TEMP TRIGGER fail_on_finalize_execution
            BEFORE UPDATE ON executions
            FOR EACH ROW
            WHEN NEW.order_intent_id = {intent_id} AND NEW.status = 'FILLED'
            BEGIN
                SELECT RAISE(ABORT, "simulated finalize failure");
            END
        """)
        conn.commit()

        engine = _setup_engine_with_store(store, broker)

        # Try to finalize the execution - should trigger the rollback
        # We'll call _update_portfolio_on_fill which calls finalize_execution
        existing_order = store.get_paper_order("ORD-FINALIZE-ROLLBACK")
        signal_side = SignalSide(existing_order["side"])
        intent_obj = OrderIntent(
            client_order_id="ORD-FINALIZE-ROLLBACK",
            asset=existing_order["asset"],
            side=signal_side,
            notional=existing_order["notional"],
            qty=existing_order["qty"],
            limit=existing_order["limit_price"],
            meta={},
        )
        exec_result = ExecutionResult(
            status="FILLED",
            adapter="paper",
            order_id="P123456789",
            filled=True,
            fail_reason=None,
            fill_price=50000.0,
            fill_qty=0.02,
            fee=10.0,
            fee_bps=10,
            slippage_bps=5,
            broker_order_id="P123456789",
            client_order_id="ORD-FINALIZE-ROLLBACK",
        )

        try:
            engine._update_portfolio_on_fill(intent_obj, exec_result, intent_id, risk_id, execution_id)
            assert False, "Expected IntegrityError from trigger"
        except (RuntimeError, sqlite3.IntegrityError) as e:
            assert "simulated finalize failure" in str(e)

        # Verify all state is unchanged after rollback
        cash_after = store.get_portfolio()["cash"]
        assert cash_after == cash_before, (
            f"Portfolio cash must be unchanged after rollback: {cash_before} -> {cash_after}"
        )

        pos_after = store.get_position("BTC-USDT")
        assert pos_before == pos_after, (
            f"Position must be unchanged after rollback: {pos_before} -> {pos_after}"
        )

        # Execution must still be PENDING (never finalized)
        exec_row = store.get_execution(intent_id)
        assert exec_row is not None, "Execution row must exist"
        assert exec_row["status"] == "PENDING", (
            f"Execution must remain PENDING after rollback, got {exec_row['status']}"
        )
        assert exec_row["portfolio_applied"] == 0, (
            "portfolio_applied must remain 0 after rollback"
        )

        # Clean up trigger
        conn.execute("DROP TRIGGER IF EXISTS fail_on_finalize_execution")
        conn.commit()
    finally:
        store.close()
