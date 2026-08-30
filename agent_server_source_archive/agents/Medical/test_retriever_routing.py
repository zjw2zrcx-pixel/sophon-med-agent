from __future__ import annotations

from pathlib import Path
import unittest

from agents.Medical.dense import DenseSearch
from agents.Medical.retriever import MedicalRetriever


ROOT = Path(__file__).resolve().parents[2]


class _EmptyDense:
    def search(self, query: str, top_k: int = 30) -> DenseSearch:
        del query, top_k
        return DenseSearch(hits=(), embedding_ms=1.0, search_ms=2.0)


class MedicalRoutingRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = MedicalRetriever.__new__(MedicalRetriever)

    def test_department_intent_precedes_symptom_consultation(self) -> None:
        matches = ({"label": "症状"},)
        self.assertEqual(
            self.retriever._detect_intent("肚子痛应该挂什么科", matches),
            "department_recommendation",
        )

    def test_negated_fever_is_not_positive_or_reasked(self) -> None:
        positive, negative = self.retriever._symptom_polarity(
            "我没有发烧，就是头晕"
        )
        self.assertEqual(positive, ["头晕"])
        self.assertEqual(negative, ["发烧"])
        questions = self.retriever._clarifying_questions(
            "我没有发烧，就是头晕", (), "symptom_consultation", False, negative
        )
        self.assertFalse(any("发烧" in item or "发热" in item for item in questions))

    def test_cardiac_combination_and_negation(self) -> None:
        query = "胸口闷，左胳膊还有点麻"
        _, negative = self.retriever._symptom_polarity(query)
        flags = self.retriever._red_flags(query, negative)
        self.assertIn("cardiac_symptom_combination", [item["code"] for item in flags])

        negated = "我没有胸口闷，只是左胳膊有点麻"
        _, negative = self.retriever._symptom_polarity(negated)
        self.assertFalse(self.retriever._red_flags(negated, negative))

    def test_pediatric_persistent_fever_rash_is_urgent(self) -> None:
        query = "孩子高烧三天又出了皮疹"
        _, negative = self.retriever._symptom_polarity(query)
        flags = self.retriever._red_flags(query, negative)
        self.assertIn(
            "pediatric_persistent_fever_rash", [item["code"] for item in flags]
        )

    def test_explicit_aspect_patterns_override_ambiguous_entity_labels(self) -> None:
        ambiguous = ({"label": "症状"},)
        cases = {
            "引起高血压肾病的原因": "causes",
            "妇女得了霉菌性阴道炎怎么办": "treatment",
            "盆腔脓肿从B超可以看出吗": "checks",
            "怎样缓解尿道炎疼痛": "treatment",
            "女孩长期闭经会怎么样": "complications",
            "脚上长痤疮是什么样的": "symptoms",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.retriever._detect_intent(query, ambiguous), expected)

    def test_specific_fact_aspects_precede_generic_or_entity_wording(self) -> None:
        ambiguous = ({"label": "症状"},)
        cases = {
            "如何预防哮喘": "prevention",
            "哮喘有哪些风险因素": "risk_factors",
            "本地资料里有哮喘的手术治疗方法吗": "surgery",
            "术后感染可能有哪些并发症": "complications",
            "治疗后症状有哪些风险因素": "risk_factors",
            "新诊断ITP通常如何治疗": "treatment",
            "阿片类药物成瘾通常有哪些症状": "symptoms",
            "本地资料里有药物导致的肺部疾病的手术治疗方法吗": "surgery",
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(self.retriever._detect_intent(query, ambiguous), expected)

    def test_non_medical_and_navigation_requests_are_gated(self) -> None:
        self.assertEqual(
            self.retriever._non_medical_reason("水在标准大气压下沸腾温度是多少"),
            "general_knowledge_request",
        )
        self.assertEqual(
            self.retriever._non_medical_reason("买药18块，绷带7块，我拿50块，该找多少？"),
            "arithmetic_request",
        )
        self.assertEqual(
            self.retriever._non_medical_reason("抽血的地方怎么走？"),
            "navigation_request",
        )

    def test_focus_matches_removes_duplicate_surface_and_generic_entities(self) -> None:
        matches = (
            {"id": 1, "name": "卵巢囊肿", "surface": "卵巢囊肿", "label": "症状", "match": "exact", "confidence": 1.0},
            {"id": 2, "name": "卵巢囊肿", "surface": "卵巢囊肿", "label": "疾病", "match": "exact", "confidence": 1.0},
            {"id": 3, "name": "手术", "surface": "手术", "label": "治疗方案", "match": "exact", "confidence": 1.0},
        )
        focused = self.retriever._focus_matches(matches)
        self.assertEqual([(item["name"], item["label"]) for item in focused], [("卵巢囊肿", "疾病")])

    def test_navigation_side_task_is_removed_from_medical_query(self) -> None:
        query, side_task = self.retriever._medical_subquery(
            "痤疮怎么治疗，另外请带我去门诊大厅。"
        )
        self.assertEqual(query, "痤疮怎么治疗")
        self.assertIn("门诊大厅", side_task)


class MedicalRoutingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = MedicalRetriever(ROOT / "med_database" / "med_search.sqlite")

    def test_cold_department_query_returns_sourced_respiratory_department(self) -> None:
        result = self.retriever.consult("有点感冒，还未挂号，我该去哪")
        self.assertEqual(result["intent"], "department_recommendation")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["departments"][0]["name"], "呼吸内科")
        self.assertEqual(result["departments"][0]["basis"], "direct_disease")
        self.assertTrue(result["departments"][0]["source"])

    def test_symptom_consultation_runs_document_hybrid_when_dense_is_available(self) -> None:
        self.retriever._dense_retriever = _EmptyDense()
        try:
            result = self.retriever.consult("我没有发烧，就是头晕，站起来不稳")
        finally:
            self.retriever._dense_retriever = None
        self.assertEqual(result["intent"], "symptom_consultation")
        self.assertEqual(result["status"], "need_more_info")
        self.assertTrue(result["retrieval"]["dense_used"])
        self.assertEqual(result["retrieval"]["mode"], "hybrid")
        self.assertEqual(result["associations"], [])

    def test_cause_query_returns_only_cause_aspects(self) -> None:
        result = self.retriever.consult("引起高血压肾病的原因")
        self.assertEqual(result["intent"], "causes")
        self.assertEqual(result["associations"], [])
        facts = [item for item in result["evidence"] if item["type"] == "fact"]
        self.assertTrue(facts)
        self.assertTrue(all(item["aspect"] in {"病因", "发病机制", "遗传因素", "传播途径"} for item in facts))

    def test_new_specific_intents_return_only_the_requested_fact_aspect(self) -> None:
        cases = {
            "如何预防哮喘": ("prevention", "预防"),
            "哮喘有哪些风险因素": ("risk_factors", "风险因素"),
            "本地资料里有哮喘的手术治疗方法吗": ("surgery", "手术治疗"),
        }
        for query, (intent, aspect) in cases.items():
            with self.subTest(query=query):
                result = self.retriever.consult(query)
                self.assertEqual(result["intent"], intent)
                facts = [
                    item for item in result["evidence"]
                    if item["type"] == "fact"
                ]
                self.assertTrue(facts)
                self.assertTrue(all(item["aspect"] == aspect for item in facts))

    def test_query_prefix_recovers_fact_subject_lost_by_graph_normalization(self) -> None:
        for query, aspect in (
            ("如何预防1型糖尿病", "预防"),
            ("1型糖尿病有哪些风险因素", "风险因素"),
        ):
            with self.subTest(query=query):
                result = self.retriever.consult(query)
                facts = [
                    item for item in result["evidence"]
                    if item["type"] == "fact"
                ]
                self.assertTrue(facts)
                self.assertEqual(facts[0]["subject"], "1型糖尿病")
                self.assertEqual(facts[0]["aspect"], aspect)

    def test_named_entity_without_requested_aspect_returns_explicit_gap(self) -> None:
        result = self.retriever.consult(
            "本地资料里有老年人脂肪肝的手术治疗方法吗"
        )
        self.assertEqual(result["intent"], "surgery")
        self.assertEqual(result["status"], "evidence_gap")
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["associations"], [])
        self.assertIn("没有覆盖当前所问方面", result["message"])

    def test_generic_surgery_entity_does_not_expand_to_unrelated_diseases(self) -> None:
        result = self.retriever.consult("卵巢囊肿手术后治疗方案")
        self.assertEqual(result["intent"], "surgery")
        self.assertNotIn("手术", [item["canonical"] for item in result["normalized_terms"]])
        self.assertFalse(any(item["related"] in {"肌张力异常", "包涵囊肿", "上颌窦炎"} for item in result["associations"]))

    def test_complication_query_excludes_drugs_and_cross_disease_document(self) -> None:
        result = self.retriever.consult("锁喉毒的并发症")
        self.assertTrue(all(item["relation"] == "并发症" for item in result["associations"]))
        self.assertFalse(any("毒气中毒" in item.get("question", "") for item in result["evidence"]))

    def test_out_of_scope_request_skips_all_retrieval(self) -> None:
        result = self.retriever.consult("水在标准大气压下沸腾温度是多少")
        self.assertEqual(result["status"], "out_of_scope")
        self.assertEqual(result["intent"], "non_medical")
        self.assertEqual(result["retrieval"]["mode"], "not_run")
        self.assertEqual(result["evidence"], [])

    def test_mixed_navigation_clause_does_not_pollute_medical_entities(self) -> None:
        result = self.retriever.consult("痤疮怎么治疗，另外请带我去门诊大厅。")
        self.assertEqual(result["intent"], "treatment")
        self.assertEqual(result["retrieval"]["medical_query"], "痤疮怎么治疗")
        self.assertNotIn("门诊", [item["canonical"] for item in result["normalized_terms"]])


if __name__ == "__main__":
    unittest.main()
