"""
Payment/Reward models — spec 5.1.2 session:reward + session:payment.
"""

from typing import Any, Optional
from pydantic import BaseModel


class RewardData(BaseModel):
    """S2C session:reward payload.data"""
    payment_id: Optional[str] = None
    content: Optional[str] = None
    charge_type: str = ""
    currency_code: Optional[str] = None
    currency_symbol: Optional[str] = None
    original_bill_amount: Optional[float] = None
    original_bill_amount_usd: Optional[float] = None
    estimated_savings: Optional[float] = None
    estimated_savings_usd: Optional[float] = None
    charge_percentage: Optional[float] = None
    charge_percentage_min: Optional[float] = None
    charge_percentage_tips: Optional[list[float]] = None
    fixed_charge_amount: Optional[float] = None
    fixed_charge_amount_min: Optional[float] = None
    fixed_charge_amount_tips: Optional[list[float]] = None
    status: Optional[str] = None
    stripe_intent_id: Optional[str] = None
    stripe_client_secret: Optional[str] = None
    coupon_id: Optional[str] = None
    coupon_amount: Optional[float] = None
    reject_status: Optional[str] = None


class PaymentData(BaseModel):
    """S2C session:payment payload.data"""
    payment_id: Optional[str] = None
    message: Optional[str] = None
    charge_type: str = ""
    currency_code: Optional[str] = None
    currency_symbol: Optional[str] = None
    original_bill_amount: Optional[float] = None
    original_bill_amount_usd: Optional[float] = None
    estimated_savings: Optional[float] = None
    estimated_savings_usd: Optional[float] = None
    actual_savings: Optional[float] = None
    actual_savings_usd: Optional[float] = None
    actual_payment_amount: Optional[float] = None
    service_fee_collected: Optional[float] = None
    service_fee_refunded: Optional[float] = None
    coupon_amount: Optional[float] = None
    status: Optional[str] = None
    task_status: Optional[str] = None
    cancel_reason: Optional[str] = None
