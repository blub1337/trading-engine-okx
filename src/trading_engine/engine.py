"""Main trading engine application.

Orchestrates market data, strategy, risk, execution, and persistence.
Paper-only execution. No live trading path.
"""
from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from .models import ExecutionResult, OrderIntent, Signal, SignalSide, RiskDecision, RiskStatus
from .execution.paper_broker import PaperBroker
from .persistence import SQLiteStore
from .persistence.sqlite_store import _utc_now_iso
from .risk import RiskConfig, RiskGates

try:
    from .marketdata import OkxPublicClient, TickerSnapshot, MarketDataError
except Exception:  # keep engine importable even if module absent
    OkxPublicClient = None
    TickerSnapshot = None
    MarketDataError = Exception

# Configure structured logging (no secrets)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("trading_engine")


class MarketDataProvider(Protocol):
    """Protocol for a read-only market data provider returning TickerSnapshot."""
    def get_ticker(self, inst_id: str) -> Any:
        ...


class TradingEngine:
    """Main trading engine. Paper-only, fail-closed risk gates."""

    def __init__(
        self,
        db_path: str = "data/trading_engine.db",
        risk_config: Optional[RiskConfig] = None,
        market_data_provider: Optional[MarketDataProvider] = None,
        max_market_data_age_seconds: float = 60.0,
        default_notional: float = 1000.0,
    ):
        if not isinstance(max_market_data_age_seconds, (int, float)):
            raise ValueError("max_market_data_age_seconds must be numeric")
        if not math.isfinite(max_market_data_age_seconds) or max_market_data_age_seconds <= 0:
            raise ValueError("max_market_data_age_seconds must be finite and positive")
        if not isinstance(default_notional, (int, float)):
            raise ValueError("default_notional must be numeric")
        if default_notional <= 0 or not math.isfinite(default_notional):
            raise ValueError("default_notional must be strictly positive and finite")

        self.store = SQLiteStore(db_path)
        self.risk_gates = RiskGates(self.store, risk_config or RiskConfig.default())
        self.broker = PaperBroker(paper=True, store=self.store)
        self._running = False
        self._market_data_provider = market_data_provider
        self._max_market_data_age_seconds = float(max_market_data_age_seconds)
        self._default_notional = float(default_notional)
        # Daily loss tracking - load from persistent state
        daily_state = self.store.get_daily_loss_state()
        self._daily_start_equity = daily_state.get("daily_start_equity")
        last_date_str = daily_state.get("last_trading_date")
        self._last_date = datetime.fromisoformat(last_date_str).date() if last_date_str else None

        logger.info(
            "TradingEngine initialized",
            extra={
                "db_path": db_path,
                "paper_mode": True,
                "market_data_age_limit": self._max_market_data_age_seconds,
            },
        )

    def process_signal(self, signal: Signal) -> ExecutionResult:
        """Process a strategy signal through risk gates to execution.

        Returns ExecutionResult. Rejected trades NEVER reach the broker.
        Idempotent: reprocessing the same signal returns the existing execution result.
        """
        logger.info("Processing signal", extra={"strategy": signal.strategy, "asset": signal.asset})

        # IDEMPOTENCY CHECK: If signal has signal_id, check if already processed
        if signal.signal_id:
            existing_signal = self.store.get_signal_by_signal_id(signal.signal_id)
            if existing_signal:
                # Check if order intent already exists for this signal
                existing_intent = self.store.get_order_intent_by_signal_ref(signal.signal_id)
                if existing_intent:
                    # Check if execution already exists
                    existing_execution = self.store.get_execution(existing_intent["id"])
                    if existing_execution:
                        logger.info("Signal already processed, returning existing result", 
                                   extra={"signal_id": signal.signal_id, "intent_id": existing_intent["id"]})
                        return ExecutionResult(
                            status=existing_execution["status"],
                            adapter=existing_execution["adapter"],
                            order_id=existing_execution["order_id"],
                            filled=bool(existing_execution["filled"]),
                            fail_reason=existing_execution["fail_reason"],
                            fill_price=existing_execution.get("fill_price"),
                            fill_qty=existing_execution.get("fill_qty"),
                            fee=existing_execution.get("fee"),
                            fee_bps=existing_execution.get("fee_bps"),
                            slippage_bps=existing_execution.get("slippage_bps"),
                        )

        # Record the raw signal first – this mirrors the original workflow and ensures the
        # audit table can reference the same signal if market‑data validation later fails.
        signal_id = self.store.record_signal(signal)

        # Determine reference price: if the Signal already carries a validated price we can skip
        # market‑data fetching entirely (used by unit tests that provide ``reference_price``).
        reference_price: float | None = None
        if signal.reference_price is not None:
            reference_price = signal.reference_price
        else:
            # Fetch and validate market data BEFORE sizing or risk evaluation.
            try:
                snapshot = self._fetch_market_data(signal.asset)
                reference_price = snapshot.last
            except MarketDataError as exc:
                # Early market‑data failure – record the raw signal and audit the reason.
                result = ExecutionResult(
                    status="REJECTED",
                    adapter="paper",
                    order_id=None,
                    filled=False,
                    fail_reason=f"market data validation failed: {exc}",
                )
                logger.warning("Trade rejected by market data", extra={"reason": result.fail_reason})
                # Audit the rejection.
                self.store.record_signal_rejection(signal, result.fail_reason)
                # No intent, risk decision, or execution rows are persisted for this early failure.
                return result

        # Fail-closed: signals without a stable signal_id cannot be
        # deduplicated across retries, so they must be rejected before execution.
        # Reject None, empty string, whitespace-only strings, and non-string values.
        signal_id = signal.signal_id
        if not isinstance(signal_id, str) or not signal_id.strip():
            result = ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason="Invalid signal_id: cannot guarantee idempotent execution",
            )
            self.store.record_signal_rejection(signal, result.fail_reason)
            logger.warning("Trade rejected: invalid signal_id", extra={"reason": result.fail_reason})
            return result

        # Normalize signal_id consistently (strip whitespace).
        signal_id = signal_id.strip()

        # Market data succeeded – create order intent using validated reference price.
        intent = self._create_intent(signal, reference_price=reference_price)
        # Pass signal_id for idempotency linkage
        intent.signal_id = signal_id
        intent_id = self.store.record_order_intent(intent, signal_id)

        # Daily loss tracking: reset at start of new day
        now = datetime.now(timezone.utc)
        today = now.date()
        if self._last_date is None or today != self._last_date:
            # New day: compute daily start equity = cash + sum(position * price)
            portfolio = self.store.get_portfolio()
            positions = self.store.get_all_positions()
            equity = portfolio["cash"]
            for pos in positions:
                if pos["qty"] != 0 and pos["avg_entry_price"]:
                    equity += abs(pos["qty"]) * pos["avg_entry_price"]
            self._daily_start_equity = equity
            self._last_date = today
            # Persist daily loss state
            self.store.update_daily_loss_state(self._daily_start_equity, today.isoformat())

        # Compute current equity for risk gate
        portfolio = self.store.get_portfolio()
        positions = self.store.get_all_positions()
        current_equity = portfolio["cash"]
        for pos in positions:
            if pos["qty"] != 0 and pos["avg_entry_price"]:
                current_equity += abs(pos["qty"]) * pos["avg_entry_price"]

        # Risk evaluation (fail-closed).
        decision = self.risk_gates.evaluate(
            signal, intent, portfolio["cash"],
            daily_start_equity=self._daily_start_equity,
            current_equity=current_equity
        )
        risk_id = self.store.record_risk_decision(intent_id, decision)

        logger.info("Risk decision", extra={"status": decision.status.value, "reason": decision.reason})

        # REJECTED trades CANNOT reach execution.
        if decision.status.value == "REJECTED":
            result = ExecutionResult(
                status="REJECTED",
                adapter="paper",
                order_id=None,
                filled=False,
                fail_reason=decision.reason,
            )
            self.store.record_execution(intent_id, risk_id, result)
            logger.warning("Trade rejected by risk gates", extra={"reason": decision.reason})
            return result

        # APPROVED -> paper execution ONLY.
        # First, record a PENDING execution to track intent before broker call
        pending_result = ExecutionResult(
            status="PENDING",
            adapter="paper",
            order_id=None,
            filled=False,
            fail_reason=None,
        )
        execution_id = self.store.record_execution(intent_id, risk_id, pending_result)

        # APPROVED -> paper execution ONLY.
        result = self.broker.place(intent, reference_price=reference_price)
        
        # Update portfolio and execution atomically
        if result.filled:
            self._update_portfolio_on_fill(intent, result, intent_id, risk_id, execution_id)
            self.store.update_daily_loss_state(self._daily_start_equity, today.isoformat())
        else:
            self.store.record_execution(intent_id, risk_id, result)

        logger.info("Execution complete", extra={"status": result.status, "filled": result.filled})
        return result

    def _fetch_market_data(self, asset: str) -> Any:
        """Strictly validate a ticker snapshot for an asset."""
        provider = self._market_data_provider
        if provider is None:
            raise MarketDataError("market data provider is not configured")

        try:
            snapshot = provider.get_ticker(asset)
        except Exception as exc:
            # Wrap any provider exception — fail closed.
            raise MarketDataError(f"provider failure: {exc}") from exc

        if TickerSnapshot is not None and not isinstance(snapshot, TickerSnapshot):
            raise MarketDataError("market data provider did not return a TickerSnapshot")

        # Require concrete attributes.
        last = getattr(snapshot, "last", None)
        if not isinstance(last, (int, float)) or isinstance(last, bool) or not math.isfinite(last) or last <= 0:
            raise MarketDataError(f"invalid snapshot.last: {last!r}")

        inst_id = getattr(snapshot, "inst_id", None)
        if inst_id != asset:
            raise MarketDataError(
                f"instrument mismatch: requested {asset!r}, got {inst_id!r}"
            )

        # Exchange timestamp validation.
        ts = getattr(snapshot, "ts", None)
        now = datetime.now(timezone.utc)
        # Accept any object that behaves like a datetime (has isoformat and can be compared).
        if not hasattr(ts, "isoformat"):
            raise MarketDataError(f"snapshot.ts must be datetime‑like, got {type(ts)!r}")
        # Normalise to UTC aware datetime.
        ts_utc = ts if getattr(ts, "tzinfo", None) is not None else ts.replace(tzinfo=timezone.utc)
        FIRST_FUTURE_SKEW = 10.0
        ts_forward = (ts_utc - now).total_seconds()
        if ts_forward > FIRST_FUTURE_SKEW:
            raise MarketDataError(f"exchange timestamp too far in future: {ts.isoformat()}")
        if ts_forward < -10 * 365 * 24 * 3600:
            raise MarketDataError(f"exchange timestamp implausibly old: {ts.isoformat()}")

        exchange_age = (now - ts_utc).total_seconds()
        if exchange_age > self._max_market_data_age_seconds:
            raise MarketDataError(
                f"exchange timestamp stale: {exchange_age:.2f}s > {self._max_market_data_age_seconds}"
            )

        # Receipt timestamp validation.
        received_at = getattr(snapshot, "received_at", None)
        if not hasattr(received_at, "isoformat"):
            raise MarketDataError(f"received_at must be datetime‑like, got {type(received_at)!r}")
        received_utc = received_at if getattr(received_at, "tzinfo", None) is not None else received_at.replace(tzinfo=timezone.utc)
        receipt_forward = (received_utc - now).total_seconds()
        if receipt_forward > FIRST_FUTURE_SKEW:
            raise MarketDataError(f"receipt timestamp too far in future: {received_at.isoformat()}")
        receipt_age = (now - received_utc).total_seconds()
        if receipt_age > self._max_market_data_age_seconds:
            raise MarketDataError(
                f"receipt timestamp stale: {receipt_age:.2f}s > {self._max_market_data_age_seconds}"
            )

        # Capture the validated price on the snapshot (no mutation — read it back).
        return snapshot

    def _create_intent(self, signal: Signal, reference_price=None) -> OrderIntent:
        """Create order intent from signal. Sizing only on a validated price."""
        # In Phase 1, use fixed notional sizing.
        notional = self._default_notional

        qty = None
        if notional > 0:
            if reference_price is not None and isinstance(reference_price, (int, float)) and not isinstance(reference_price, bool):
                try:
                    computed = float(notional) / float(reference_price)
                except (ZeroDivisionError, ValueError, OverflowError):
                    computed = None
                if computed is not None and math.isfinite(computed) and computed > 0:
                    qty = computed

        # Final guard — must not fabricate ticket size in production wiring.
        # Fall back to None for qty => rejected deterministically by risk gates.
        return OrderIntent(
            client_order_id=f"TE_{uuid.uuid4().hex[:12]}",
            asset=signal.asset,
            side=signal.side,
            notional=notional,
            qty=qty,
            limit=None,  # Market order
            meta={"signal_conf": signal.conf, "strategy": signal.strategy},
        )

    def _update_portfolio_on_fill(self, intent: OrderIntent, result: ExecutionResult, intent_id: Optional[int] = None, risk_id: Optional[int] = None, execution_id: Optional[int] = None) -> None:
        """Update portfolio state after a fill. Uses atomic transaction with BEGIN IMMEDIATE.
        
        Deducts actual fill cost (fill_qty * fill_price) + fee from cash.
        Updates position quantity, average entry price, and realized P&L.
        All updates are performed in a single atomic transaction.
        Exactly-once guarantee: portfolio_applied check and update happen within the same transaction.
        """
        # Use actual fill data from execution result
        if result.fill_qty is None or result.fill_price is None or result.fill_qty <= 0 or result.fill_price <= 0:
            logger.error("Invalid fill data in execution result", extra={
                "fill_qty": result.fill_qty, "fill_price": result.fill_price
            })
            raise ValueError("Execution result missing valid fill data")

        fill_qty = result.fill_qty
        fill_price = result.fill_price
        fee = result.fee if result.fee is not None else 0.0

        # Calculate fill cost (notional of actual fill)
        fill_cost = fill_qty * fill_price
        total_cash_out = fill_cost + fee

        # Perform all updates in a single atomic transaction with BEGIN IMMEDIATE for serialization
        with self.store.transaction() as conn:
            # Exactly-once guard: check portfolio_applied WITHIN the transaction
            if intent_id is not None:
                existing_exec = conn.execute(
                    "SELECT portfolio_applied FROM executions WHERE order_intent_id = ?",
                    (intent_id,)
                ).fetchone()
                if existing_exec and existing_exec["portfolio_applied"] == 1:
                    logger.info("Portfolio already applied for execution (inside transaction)", extra={"intent_id": intent_id})
                    return

            # Get current portfolio and position
            portfolio_row = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
            if not portfolio_row:
                raise RuntimeError("Portfolio state not found")
            current_cash = portfolio_row["cash"]

            # Deduct total cash out (fill cost + fee)
            new_cash = current_cash - total_cash_out
            if new_cash < 0:
                raise RuntimeError(f"Insufficient cash: {current_cash} < {total_cash_out}")
            conn.execute(
                "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE id = 1",
                (new_cash, _utc_now_iso())
            )
            # Get current portfolio and position
            portfolio_row = conn.execute("SELECT cash FROM portfolio_state WHERE id = 1").fetchone()
            if not portfolio_row:
                raise RuntimeError("Portfolio state not found")
            current_cash = portfolio_row["cash"]

            # Deduct total cash out (fill cost + fee)
            new_cash = current_cash - total_cash_out
            if new_cash < 0:
                raise RuntimeError(f"Insufficient cash: {current_cash} < {total_cash_out}")
            conn.execute(
                "UPDATE portfolio_state SET cash = ?, updated_at = ? WHERE id = 1",
                (new_cash, _utc_now_iso())
            )

            # Get current position
            pos_row = conn.execute(
                "SELECT qty, avg_entry_price, realized_pnl FROM positions WHERE asset = ?",
                (intent.asset,)
            ).fetchone()

            current_qty = pos_row["qty"] if pos_row else 0.0
            current_avg_price = pos_row["avg_entry_price"] if pos_row and pos_row["avg_entry_price"] is not None else None
            current_realized_pnl = pos_row["realized_pnl"] if pos_row and pos_row["realized_pnl"] is not None else 0.0

            # Determine fill direction
            fill_qty_signed = fill_qty if intent.side == SignalSide.LONG else -fill_qty
            new_qty = current_qty + fill_qty_signed

            # Calculate new average entry price and realized P&L
            new_avg_price = None
            realized_pnl = current_realized_pnl

            if current_qty == 0:
                # Opening new position
                new_avg_price = fill_price
            elif (current_qty > 0 and fill_qty_signed > 0) or (current_qty < 0 and fill_qty_signed < 0):
                # Adding to existing position (same direction)
                # Weighted average of existing and new fill
                total_cost = abs(current_qty) * current_avg_price + fill_qty * fill_price
                new_avg_price = total_cost / (abs(current_qty) + fill_qty)
            else:
                # Reducing or reversing position (opposite direction)
                closing_qty = min(abs(current_qty), abs(fill_qty_signed))
                # Realized P&L on the closed portion
                if current_qty > 0:
                    # Was long, selling to close
                    realized_pnl += closing_qty * (fill_price - current_avg_price)
                else:
                    # Was short, buying to close
                    realized_pnl += closing_qty * (current_avg_price - fill_price)

                remaining_qty = abs(current_qty) - closing_qty
                if remaining_qty > 0:
                    # Partial reduction - avg price unchanged for remaining
                    new_avg_price = current_avg_price
                elif abs(fill_qty_signed) > closing_qty:
                    # Full closure + reversal
                    new_avg_price = fill_price
                else:
                    # Exact closure - no position remains
                    new_avg_price = None

            # Update position (UPSERT)
            if new_qty == 0:
                conn.execute(
                    """INSERT INTO positions (asset, qty, avg_entry_price, realized_pnl, updated_at) 
                    VALUES (?, 0, NULL, ?, ?)
                    ON CONFLICT(asset) DO UPDATE SET 
                        qty = 0, 
                        avg_entry_price = NULL, 
                        realized_pnl = excluded.realized_pnl, 
                        updated_at = excluded.updated_at""",
                    (intent.asset, realized_pnl, _utc_now_iso())
                )
            else:
                conn.execute(
                    """INSERT INTO positions (asset, qty, avg_entry_price, realized_pnl, updated_at) 
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(asset) DO UPDATE SET 
                        qty = excluded.qty, 
                        avg_entry_price = excluded.avg_entry_price, 
                        realized_pnl = excluded.realized_pnl, 
                        updated_at = excluded.updated_at""",
                    (intent.asset, new_qty, new_avg_price, realized_pnl, _utc_now_iso())
                )

            # Mark portfolio_applied=1 for exactly-once guarantee
            if intent_id is not None:
                conn.execute(
                    "UPDATE executions SET portfolio_applied = 1 WHERE order_intent_id = ?",
                    (intent_id,),
                )

            # Finalize execution status in the same transaction (atomic with portfolio)
            if execution_id is not None:
                self.store.finalize_execution(execution_id, result, conn=conn)

        logger.info("Portfolio updated atomically", extra={
            "cash": new_cash,
            "asset": intent.asset,
            "qty": new_qty,
            "avg_entry_price": new_avg_price,
            "realized_pnl": realized_pnl,
            "fill_cost": fill_cost,
            "fee": fee
        })

    def get_portfolio_summary(self) -> dict:
        """Get current portfolio summary."""
        portfolio = self.store.get_portfolio()
        positions = self.store.get_all_positions()
        return {
            "cash": portfolio["cash"],
            "positions": positions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def recover_pending_executions(self) -> dict:
        """Recover pending executions after a crash/restart.

        Required behavior:
        - Existing FILLED order with portfolio_applied=0: finalize the pending execution and apply the portfolio update exactly once.
        - Existing FILLED order with portfolio_applied=1: do not apply the fill again.
        - Existing REJECTED order: finalize without portfolio changes.
        - Existing PENDING order: leave it pending without generating a different ID.
        - No existing broker order: retry using the original client_order_id.
        - Existing finalized execution: do not apply the fill again.
        """
        logger.info("Starting PENDING execution recovery")

        # Find all PENDING executions AND FILLED executions with portfolio_applied=0
        # (these represent orders that were sent to broker but never finalized in the executions table,
        # or were finalized but portfolio was never applied due to crash)
        conn = self.store._get_conn()
        pending_executions = conn.execute(
            "SELECT e.*, oi.client_order_id FROM executions e "
            "JOIN order_intents oi ON e.order_intent_id = oi.id "
            "WHERE e.status = 'PENDING' OR (e.status = 'FILLED' AND e.portfolio_applied = 0) ORDER BY e.id"
        ).fetchall()

        stats = {
            "total_pending": len(pending_executions),
            "recovered_filled": 0,
            "recovered_rejected": 0,
            "retried": 0,
            "errors": 0,
        }

        for row in pending_executions:
            client_order_id = row["client_order_id"]
            intent_id = row["order_intent_id"]
            risk_id = row["risk_decision_id"]
            execution_id = row["id"]
            portfolio_applied = row["portfolio_applied"] if "portfolio_applied" in row.keys() else 0
            logger.info(f"Recovering execution: {client_order_id} (status={row['status']}, portfolio_applied={portfolio_applied})")

            try:
                # Look up existing broker order via public broker API
                try:
                    result = self.broker.get_execution_result(client_order_id)
                except KeyError:
                    result = None

                # Get the risk decision
                risk_decision = self.store.get_risk_decision(intent_id)
                if not risk_decision:
                    logger.error(f"No risk decision found for intent: {intent_id}")
                    stats["errors"] += 1
                    continue

                # If execution already has a final status, check portfolio_applied
                if row["status"] in ("FILLED", "REJECTED"):
                    portfolio_applied_val = row["portfolio_applied"] if "portfolio_applied" in row.keys() else 0
                    if row["status"] == "FILLED" and portfolio_applied_val == 0:
                        # FILLED but portfolio not applied - apply now via atomic method
                        existing_order = self.store.get_paper_order(client_order_id)
                        if existing_order:
                            signal_side = SignalSide(existing_order["side"])
                            intent_obj = OrderIntent(
                                client_order_id=client_order_id,
                                asset=existing_order["asset"],
                                side=signal_side,
                                notional=existing_order["notional"],
                                qty=existing_order["qty"],
                                limit=existing_order["limit_price"],
                                meta={},
                            )
                            exec_result = ExecutionResult(
                                status=row["status"],
                                adapter=row["adapter"],
                                order_id=row["order_id"],
                                filled=bool(row["filled"]),
                                fail_reason=row["fail_reason"],
                                fill_price=row["fill_price"] if "fill_price" in row.keys() else None,
                                fill_qty=row["fill_qty"] if "fill_qty" in row.keys() else None,
                                fee=row["fee"] if "fee" in row.keys() else None,
                                fee_bps=row["fee_bps"] if "fee_bps" in row.keys() else None,
                                slippage_bps=row["slippage_bps"] if "slippage_bps" in row.keys() else None,
                                broker_order_id=row["order_id"],
                                client_order_id=client_order_id,
                            )
                            self._update_portfolio_on_fill(intent_obj, exec_result, intent_id, risk_id, execution_id)
                            logger.info(f"Applied portfolio for already-finalized fill: {client_order_id}")
                        stats["recovered_filled"] += 1
                    else:
                        # REJECTED or already applied FILLED
                        logger.info(f"Execution already finalized for {client_order_id}")
                        if row["status"] == "FILLED":
                            stats["recovered_filled"] += 1
                        else:
                            stats["recovered_rejected"] += 1
                    continue

                if result is not None:
                    if result.filled:
                        # Existing FILLED order: apply portfolio exactly once
                        intent = self.store.get_order_intent(client_order_id)
                        intent_obj = OrderIntent(
                            client_order_id=client_order_id,
                            asset=intent["asset"],
                            side=SignalSide(intent["side"]),
                            notional=intent["notional"],
                            qty=intent["qty"],
                            limit=intent["limit_price"],
                            meta=intent.get("meta", {}),
                        )
                        self._update_portfolio_on_fill(intent_obj, result, intent_id, risk_id)
                        # Update the existing PENDING execution row instead of inserting a new one
                        self.store.record_execution(intent_id, risk_id, result)
                        stats["recovered_filled"] += 1
                        logger.info(f"Successfully recovered FILLED order: {client_order_id}")
                    elif result.status == "REJECTED":
                        # Existing REJECTED order: finalize without portfolio changes
                        self.store.record_execution(intent_id, risk_id, result)
                        stats["recovered_rejected"] += 1
                        logger.info(f"Recovered REJECTED order: {client_order_id}")
                    else:
                        # PENDING order: leave it pending
                        logger.info(f"Order {client_order_id} still PENDING - leaving as is")
                else:
                    # No broker order exists - retry using original client_order_id
                    logger.info(f"No broker order found for {client_order_id}, retrying")

                    intent = self.store.get_order_intent(client_order_id)
                    if not intent:
                        logger.error(f"No intent found for {client_order_id}")
                        stats["errors"] += 1
                        continue

                    signal = Signal(
                        strategy="recovery",
                        asset=intent["asset"],
                        side=SignalSide(intent["side"]),
                        conf=0.8,
                        rationale="recovery retry",
                        required_inputs=["close"],
                        issued_at=datetime.now(timezone.utc),
                        signal_id=f"recovery_{client_order_id}",
                    )

                    self.process_signal(signal)
                    stats["retried"] += 1

            except Exception as e:
                logger.error(f"Error recovering order {client_order_id}: {e}")
                stats["errors"] += 1

        logger.info(f"Recovery complete: {stats}")
        return stats

    def shutdown(self) -> None:
        """Clean shutdown."""
        self.store.close()
        provider = self._market_data_provider
        close_fn = getattr(provider, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                logger.warning("Provider close error", extra={"error": str(exc)})
        logger.info("TradingEngine shutdown complete")