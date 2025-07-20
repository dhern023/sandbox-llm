"""
Pipeline Overview
User Query
   ↓
Intent Classifier → ConditionalRouter
   ↓                     ↓
Query Expansion      Retrieval → Reranker
   ↓                     ↓
Clarification Prompt   Context Assembly
   ↓                     ↓
LLM Generation → ConditionalRouter
   ↓                     ↓
Answer or Fallback (e.g., document snippets or web search)

"""

