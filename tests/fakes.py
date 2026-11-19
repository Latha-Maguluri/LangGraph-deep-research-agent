from __future__ import annotations

import json
from types import SimpleNamespace


class DeterministicChatModel:
    def invoke(self, prompt: str, **kwargs) -> SimpleNamespace:
        if "You are a research planner for the domain" in prompt:
            content = (
                '{"summary":"Test research plan","tasks":['
                '{"title":"Market","objective":"Assess market demand","expected_evidence":"Market data"},'
                '{"title":"Competition","objective":"Assess competitors","expected_evidence":"Competitor data"},'
                '{"title":"Technology","objective":"Assess feasibility","expected_evidence":"Technical sources"},'
                '{"title":"Financial","objective":"Assess viability","expected_evidence":"Financial data"},'
                '{"title":"Risks","objective":"Assess risks","expected_evidence":"Risk sources"}]}'
            )
        elif "You are a research specialist." in prompt:
            records = json.loads(prompt.split("Retrieved evidence records:\n", 1)[1].splitlines()[0])
            content = json.dumps(
                {
                    "findings": [
                        {
                            "evidence_id": record["evidence_id"],
                            "key_finding": record["support"][:200],
                            "confidence": 0.8,
                            "conflicting_evidence_ids": [],
                            "evidence_scope": "banking",
                            "evidence_basis": "primary_empirical",
                        }
                        for record in records
                    ]
                }
            )
        elif "Create a complete inventory of every material" in prompt:
            content = '{"claims":[]}'
        elif "Material claim batch " in prompt:
            content = '{"claim_audits":[]}'
        elif "Audited material claims; every inventory item has an audit result:" in prompt:
            content = (
                '{"citation_accuracy":0.9,"research_completeness":0.9,"objectivity":0.9,'
                '"issues":[],"unsupported_claims":[],"revision_instructions":[],"claim_audits":[]}'
            )
        elif "You are an independent quality critic." in prompt:
            if "Source batch " in prompt:
                context_text = prompt.rsplit("Source batch ", 1)[1].split(":\n", 1)[1]
                audit_context = json.loads(context_text)
                candidate = next((item for item in audit_context if item.get("excerpt")), None)
                claim_audits = (
                    [
                        {
                            "claim": "Deterministic test claim.",
                            "evidence_ids": [candidate["audit_id"]],
                            "status": "supported",
                            "excerpt": candidate["excerpt"][:200],
                            "reasoning": "Deterministic test result.",
                        }
                    ]
                    if candidate
                    else []
                )
                content = json.dumps({"claim_audits": claim_audits})
                return SimpleNamespace(content=content)
            if "Batched claim-audit results:" in prompt:
                content = (
                    '{"citation_accuracy":0.9,"research_completeness":0.9,"objectivity":0.9,'
                    '"issues":[],"unsupported_claims":[],"revision_instructions":[],"claim_audits":[]}'
                )
                return SimpleNamespace(content=content)
            evidence_text = prompt.split("Evidence:\n", 1)[1].lstrip()
            audit_context, _ = json.JSONDecoder().raw_decode(evidence_text)
            candidate = next((item for item in audit_context if item.get("excerpt")), None)
            if candidate:
                claim_audits = [
                    {
                        "claim": "Deterministic test claim.",
                        "evidence_ids": [candidate["audit_id"]],
                        "status": "supported",
                        "excerpt": candidate["excerpt"][:200],
                        "reasoning": "Deterministic test result.",
                    }
                ]
            elif audit_context:
                source = audit_context[0]
                claim_audits = [
                    {
                        "claim": "Deterministic test claim.",
                        "evidence_ids": [source["audit_id"]],
                        "status": (
                            "insufficient_evidence"
                            if source["retrieval_completeness"] == "complete"
                            else "insufficient_retrieval"
                        ),
                        "excerpt": "",
                        "reasoning": "Deterministic test result.",
                    }
                ]
            else:
                claim_audits = []
            content = (
                '{"citation_accuracy":0.9,"research_completeness":0.9,"objectivity":0.9,'
                '"issues":[],"unsupported_claims":[],"revision_instructions":[],"claim_audits":'
                + json.dumps(claim_audits)
                + "}"
            )
        elif "Compare the supplied evidence records" in prompt:
            records = json.loads(prompt.split("Evidence records:\n", 1)[1])
            content = json.dumps(
                {
                    "results": [
                        {
                            "evidence_id": record["evidence_id"],
                            "status": (
                                "verified"
                                if record["retrieval_method"] == "page_excerpt"
                                else "unknown"
                            ),
                            "reasoning": "Deterministic test result.",
                        }
                        for record in records
                    ]
                }
            )
        else:
            content = "# Executive Research Report\n\n## Research Question\nTest question\n"
        return SimpleNamespace(content=content)
