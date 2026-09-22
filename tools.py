import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
FAISS_DIR = BASE_DIR / "faiss_index"
ORDER_FILE = BASE_DIR / "Order.xlsx"
PENDING_FILE = BASE_DIR / "pending_escalations.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class RAGSearchInput(BaseModel):
    query: str = Field(..., description="The customer's question or issue to search in company policies.")
    top_k: int = Field(4, description="Number of relevant policy chunks to retrieve.")


class OrderLookupInput(BaseModel):
    order_id: str = Field("", description="Order ID such as ORD-20260001. Leave blank if unknown.")
    customer_name: str = Field("", description="Customer name. Leave blank if unknown.")
    customer_email: str = Field("", description="Customer email. Leave blank if unknown.")


class EscalationInput(BaseModel):
    customer_message: str = Field(..., description="The customer's issue/request that needs human support.")
    reason: str = Field(..., description="Short reason why human support is needed.")
    conversation_summary: str = Field(..., description="Short summary for the human support team.")
    order_id: str = Field("", description="Related order ID if known.")


class PolicyRAGTool(BaseTool):
    name: str = "Company Policy Knowledge Search"
    description: str = (
        "Search the Daraz company-policy FAISS knowledge base. "
        "Use this for purchase, return, payment, ordering, shipping, refund, "
        "and customer-support policy questions. Do not invent policy facts."
    )
    args_schema: type[BaseModel] = RAGSearchInput

    def _run(self, query: str, top_k: int = 4) -> str:
        index_path = FAISS_DIR / "index.faiss"
        metadata_path = FAISS_DIR / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            return "KNOWLEDGE_BASE_UNAVAILABLE: faiss_index/index.faiss or metadata.json is missing."

        try:
            index = faiss.read_index(str(index_path))
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            model = SentenceTransformer(EMBEDDING_MODEL)
            vector = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

            k = min(max(int(top_k), 1), index.ntotal)
            scores, positions = index.search(vector, k)

            results = []
            for score, position in zip(scores[0], positions[0]):
                if position < 0 or position >= len(metadata):
                    continue

                item = metadata[position]
                results.append({
                    "score": round(float(score), 4),
                    "id": item.get("id", ""),
                    "department": item.get("department", "unknown"),
                    "source_file": item.get("source_file", "unknown"),
                    "text": item.get("text", ""),
                })

            if not results:
                return "NO_RELEVANT_POLICY_FOUND"

            return json.dumps(results, ensure_ascii=False, indent=2)

        except Exception as exc:
            return f"KNOWLEDGE_BASE_ERROR: {type(exc).__name__}: {exc}"


class OrderLookupTool(BaseTool):
    name: str = "Order Database Lookup"
    description: str = (
        "Search the simulated Order.xlsx database. "
        "Use it for order status, payment status, delivery, tracking, "
        "refund status, return eligibility, customer, seller, and order details."
    )
    args_schema: type[BaseModel] = OrderLookupInput

    def _run(
        self,
        order_id: str = "",
        customer_name: str = "",
        customer_email: str = "",
    ) -> str:
        if not ORDER_FILE.exists():
            return "ORDER_DATABASE_UNAVAILABLE: Order.xlsx is missing."

        try:
            df = pd.read_excel(ORDER_FILE, sheet_name="Orders", dtype=str).fillna("")

            mask = pd.Series(False, index=df.index)

            if order_id.strip():
                mask |= df["Order_ID"].str.casefold().eq(order_id.strip().casefold())

            if customer_email.strip():
                mask |= df["Customer_Email"].str.casefold().eq(customer_email.strip().casefold())

            if customer_name.strip():
                mask |= df["Customer_Name"].str.casefold().eq(customer_name.strip().casefold())

            if not (order_id.strip() or customer_name.strip() or customer_email.strip()):
                return "ORDER_LOOKUP_NEEDS_IDENTIFIER: Ask for an order ID, customer email, or customer name."

            matches = df.loc[mask]

            if matches.empty:
                return "NO_ORDER_FOUND"

            # Return at most 10 rows to avoid flooding the agent context.
            records = matches.head(10).to_dict(orient="records")
            return json.dumps(records, ensure_ascii=False, indent=2)

        except Exception as exc:
            return f"ORDER_DATABASE_ERROR: {type(exc).__name__}: {exc}"


class EscalateToHumanTool(BaseTool):
    name: str = "Escalate To Human Support"
    description: str = (
        "Create a pending human-support ticket. Use this when the customer explicitly "
        "asks for a human OR when the company knowledge and order database cannot safely "
        "resolve the issue. Never tell the customer that it was escalated unless this "
        "tool succeeds."
    )
    args_schema: type[BaseModel] = EscalationInput

    def _run(
        self,
        customer_message: str,
        reason: str,
        conversation_summary: str,
        order_id: str = "",
    ) -> str:
        try:
            if PENDING_FILE.exists():
                with open(PENDING_FILE, "r", encoding="utf-8") as f:
                    pending = json.load(f)
            else:
                pending = []

            ticket = {
                "ticket_id": f"SUP-{uuid.uuid4().hex[:8].upper()}",
                "status": "Pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "order_id": order_id.strip(),
                "customer_message": customer_message.strip(),
                "reason": reason.strip(),
                "conversation_summary": conversation_summary.strip(),
            }

            pending.append(ticket)

            with open(PENDING_FILE, "w", encoding="utf-8") as f:
                json.dump(pending, f, ensure_ascii=False, indent=2)

            return json.dumps({
                "success": True,
                "status": "Pending",
                "ticket_id": ticket["ticket_id"],
                "message": "Issue successfully escalated to human support."
            })

        except Exception as exc:
            return json.dumps({
                "success": False,
                "error": f"{type(exc).__name__}: {exc}"
            })


def get_pending_escalations() -> list[dict[str, Any]]:
    if not PENDING_FILE.exists():
        return []

    try:
        with open(PENDING_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []
