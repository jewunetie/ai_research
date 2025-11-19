# Literature Review: LLM Self-Compression for Question Answering

## Executive Summary

This literature review examines existing work in semantic compression, prompt compression, and question-answering evaluation methods to contextualize our research on **LLM Self-Compression for Downstream Question Answering**.

### Key Finding: **This exact experimental protocol appears to be novel.**

While there is extensive work on:
- Semantic compression with LLMs
- Prompt compression for efficiency
- Question answering from compressed contexts
- Using QA for evaluating summarization

**No prior work explicitly tests:**
1. LLMs creating **self-generated, non-human-readable** compressions
2. Using the **same model** to both create and interpret the compression
3. Evaluating these compressions via **downstream QA performance** rather than human readability or reconstruction accuracy
4. Allowing models to use **arbitrary encodings and character sets** for compression

---

## 1. Semantic Compression Methods

### 1.1 Semantic Compression with Large Language Models (2023)

**Paper:** Gilbert, H., Sandborn, M., Schmidt, D. C., Spencer-Smith, J., & White, J. (2023). *Semantic Compression With Large Language Models*. arXiv:2304.12512.

**Key Contributions:**
- Explores "approximate compression" using GPT-3.5 and GPT-4
- Introduces two novel metrics:
  - **Exact Reconstructive Effectiveness (ERE)**: Measures literal reconstruction
  - **Semantic Reconstruction Effectiveness (SRE)**: Measures semantic preservation
- Shows GPT-4 can compress and reconstruct text while preserving semantic essence
- Achieves **~5× compression ratio** while maintaining semantic content

**Difference from Our Work:**
- Focuses on **reconstruction** of the original text
- Evaluates via human judgment of semantic similarity
- Does **not** test downstream task performance (like QA)
- Assumes compression should preserve reconstructability

### 1.2 Extending Context Window via Semantic Compression (ACL 2024)

**Paper:** Chevalier, A., Wettig, A., Ajith, A., & Arora, S. (2024). *Extending Context Window of Large Language Models via Semantic Compression*. ACL 2024 Findings.

**Key Contributions:**
- Enables generalization to texts **6-8 times longer** without fine-tuning
- Semantic compression allows processing longer contexts efficiently
- No significant computational cost increase

**Difference from Our Work:**
- Goal is **context extension**, not information retention for specific tasks
- Does not evaluate via question answering
- Focuses on maintaining general coherence, not specific factual recall

---

## 2. Prompt Compression Techniques

### 2.1 LLMLingua Series (Microsoft Research, 2023-2024)

**Paper 1:** Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., & Qiu, L. (2023). *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models*. arXiv:2310.05736. EMNLP 2023.

**Paper 2:** Pan, Z., Wu, Q., Jiang, H., et al. (2024). *LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression*. arXiv:2403.12968. ACL 2024 Findings.

**Paper 3:** Jiang, H., Wu, Q., Luo, X., et al. (2023). *LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression*. arXiv:2310.06839.

**Key Contributions:**
- Achieves **up to 20x compression** with minimal performance loss
- **LLMLingua-2**: Formulates compression as token classification problem
  - Uses BERT-size model for compression
  - 3x-6x faster than previous methods
  - Accelerates end-to-end latency by 1.6x-2.9x
- **LongLLMLingua**: Specialized for long-context scenarios (RAG, QA)
- Task-agnostic and task-aware compression modes

**Difference from Our Work:**
- Goal is **computational efficiency** (reduce inference cost)
- Compression is performed by a **separate model**, not self-compression
- Compressed output is still **human-readable** (subset of original tokens)
- Not designed to test information retention limits

### 2.2 Gisting: Learning to Compress Prompts with Gist Tokens (2023)

**Paper:** Mu, J., Li, X. L., & Goodman, N. (2023). *Learning to Compress Prompts with Gist Tokens*. arXiv:2304.08467.

**Key Contributions:**
- Trains LM to compress prompts into "gist" tokens
- Gist tokens are **learned embeddings** (not text)
- Achieves **up to 26x compression** on LLaMA-7B and FLAN-T5-XXL
- No architectural modifications needed
- Training cost = standard instruction finetuning

**Recent Analysis:** Petrov, A., et al. (2025). *Long Context In-Context Compression by Getting to the Gist of Gisting*. arXiv:2504.08934.
- Shows gisting struggles with longer contexts
- Proposes **GistPool** to address limitations

**Difference from Our Work:**
- Requires **fine-tuning** the model
- Gist tokens are **continuous embeddings**, not text
- Cannot be passed as text to a new conversation
- Our approach uses zero-shot prompting with text-based compression

### 2.3 AutoCompressor: Adapting LMs to Compress Contexts (2023)

**Paper:** Chevalier, A., Wettig, A., Ajith, A., & Arora, S. (2023). *Adapting Language Models to Compress Contexts*. arXiv:2305.14788. EMNLP 2023.

**Key Contributions:**
- Compresses long contexts into **compact summary vectors** (soft prompts)
- Summary vectors are trained with unsupervised objective
- Can process sequences up to **30,720 tokens**
- Improves perplexity by utilizing long contexts
- Works well for in-context learning

**Difference from Our Work:**
- Requires **fine-tuning**
- Compression is **continuous vectors**, not discrete text
- Cannot be serialized and passed to new conversations
- Our approach is zero-shot and text-based

### 2.4 In-Context Autoencoder (ICAE) (2023)

**Paper:** Ge, T., Hu, J., Li, L., et al. (2023). *In-context Autoencoder for Context Compression in a Large Language Model*. arXiv:2307.06945.

**Key Contributions:**
- Fine-tunes LLMs with autoencoding objective
- Compresses long contexts for efficient processing
- Maintains performance on downstream tasks

**Difference from Our Work:**
- Requires fine-tuning
- Focus on efficiency rather than testing compression limits
- Does not explore non-human-readable formats

---

## 3. Context Compression for Question Answering

### 3.1 CompAct: Compressing Retrieved Documents for QA (2024)

**Paper:** Chen, J., Zhang, J., Liu, Y., & Zhang, Y. (2024). *CompAct: Compressing Retrieved Documents Actively for Question Answering*. arXiv:2407.09014.

**Key Contributions:**
- Framework for **multi-document QA**
- Sequentially compresses document segments
- Actively preserves essential information
- Designed for retrieval-augmented QA

**Relevance to Our Work:**
- Directly addresses **compression for QA**
- However, compression is guided by the question (task-aware)
- Our approach: question-agnostic compression, then answer questions from compressed form

### 3.2 CompLLM: Compression for Long Context Q&A (2025)

**Paper:** Yang, Y., et al. (2025). *CompLLM: Compression for Long Context Q&A*. arXiv:2509.19228.

**Key Contributions:**
- Focuses on long-context question answering
- Compression can be computed **offline**
- Question is **not** compressed (only context)
- Practical efficiency gains

**Relevance to Our Work:**
- Most similar to our setup: compress context, then answer questions
- However, uses standard compression techniques
- Does not explore self-generated, non-human-readable compressions
- Does not test same-model compression and interpretation

### 3.3 Recurrent Context Compression (RCC) (2024)

**Paper:** Li, Y., et al. (2024). *Recurrent Context Compression: Efficiently Expanding the Context Window of LLM*. OpenReview.

**Key Contributions:**
- Achieves **32x compression rate**
- Nearly 100% accuracy on passkey retrieval with 1M tokens
- Maintains performance on long-text QA tasks

**Difference from Our Work:**
- Focus on expanding context windows
- Not designed to test information loss characteristics

### 3.4 Selective Context (2023)

**Paper:** Li, Y., et al. (2023). *Selective Context: Compress your input to ChatGPT or other LLMs*.

**Key Contributions:**
- Compresses input to process 2x more content
- Saves 40% memory and GPU time
- Practical tool for cost reduction

**Difference from Our Work:**
- Goal is efficiency, not testing compression limits
- Does not evaluate QA performance systematically

---

## 4. Evaluation Methods: QA-Based Metrics for Summarization

### 4.1 QAEval (2021)

**Paper:** Deutsch, D., Roth, D., & Berant, J. (2021). *Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary*. Transactions of the Association for Computational Linguistics (TACL).

**Key Contributions:**
- Uses **question answering** to evaluate summary quality
- Directly measures information overlap (not text overlap)
- Achieves **state-of-the-art** correlation with human judgments
- Equals or exceeds Pyramid Method (gold standard)

**How It Works:**
1. Generate questions from reference summary
2. Answer questions using both reference and candidate summary
3. Compare answer accuracy/F1 as quality metric

**Relevance to Our Work:**
- **Exactly matches our evaluation paradigm!**
- We use QA to evaluate compression quality
- Validates that QA performance is a meaningful proxy for information retention

### 4.2 APES: Answering Performance for Evaluation of Summaries (2019)

**Paper:** Eyal, M., Baumel, T., & Elhadad, M. (2019). *Question Answering as an Automatic Evaluation Metric for News Article Summarization*. NAACL 2019.

**Key Contributions:**
- Extrinsic evaluation using reading-comprehension
- Questions focus on central entities
- Correlates with human judgment of summary quality

**Relevance to Our Work:**
- Supports using QA as compression quality metric
- Focus on factual entity information

### 4.3 SummEQuAL (2024)

**Paper:** Chen, Y., et al. (2024). *SummEQuAL: Summarization Evaluation via Question Answering using Large Language Models*. ACL 2024 NLRSE.

**Key Contributions:**
- Uses **LLMs** for QA-based evaluation
- Measures both **recall** and **precision**
- 19.7% improvement over QuestEval
- Unified, reproducible evaluation framework

**Relevance to Our Work:**
- Most recent QA-based evaluation method
- Can be adapted for our compression evaluation
- Uses modern LLMs for both QA generation and answering

---

## 5. Lossless Compression with Language Models

### 5.1 Language Modeling Is Compression (2024)

**Paper:** Delétang, G., Ruoss, A., Grau-Moya, J., et al. (2024). *Language Modeling Is Compression*. ICLR 2024. arXiv:2309.10668.

**Key Contributions:**
- Large language models are powerful general-purpose predictors
- Compression viewpoint provides insights into scaling laws, tokenization
- **Chinchilla 70B** compresses:
  - ImageNet patches to **43.4%** (vs PNG 58.5%)
  - LibriSpeech to **16.4%** (vs FLAC 30.3%)
- Shows LLMs excel at cross-domain compression

**Relevance to Our Work:**
- Establishes theoretical foundation: prediction ⟺ compression
- Shows LLMs have strong compression capabilities
- Our work extends this to **semantic** compression for QA

### 5.2 LLMZip: Lossless Text Compression (2023)

**Paper:** Valmeekam, K., Marquez, M., Olmo, A., & Kambhampati, S. (2023). *LLMZip: Lossless Text Compression using Large Language Models*. arXiv:2306.04050.

**Key Contributions:**
- Combines LLM predictions with Arithmetic Coding
- LLaMA-7B achieves **0.8426 bits/character**
- Significantly outperforms traditional compressors (gzip, bzip2)
- However, extremely slow (9.5 days for 10 MB)

**Difference from Our Work:**
- Focuses on **lossless** compression
- Goal is **perfect reconstruction**
- We focus on **lossy semantic** compression with QA evaluation

### 5.3 FineZip (2024)

**Paper:** Chen, Y., et al. (2024). *FineZip: Pushing the Limits of Large Language Models for Practical Lossless Text Compression*. arXiv:2409.17141.

**Key Contributions:**
- 54x faster than LLMZip
- Maintains high compression ratios
- More practical for real-world use

**Difference from Our Work:**
- Lossless vs. our lossy semantic compression

---

## 6. Information Bottleneck Theory

### 6.1 Deep Learning and Information Bottleneck (2015)

**Paper:** Tishby, N., & Zaslavsky, N. (2015). *Deep Learning and the Information Bottleneck Principle*. arXiv:1503.02406.

**Key Concepts:**
- Framework for understanding compression in neural networks
- Tradeoff between **accuracy** and **complexity** (compression)
- Two-phase training: fitting and compression

**Relevance to Our Work:**
- Theoretical framework for understanding compression
- Our experiment empirically tests this tradeoff in LLMs
- Can we compress while preserving task-relevant information?

### 6.2 Recent Work on Information Bottleneck in Deep Learning

**Papers:**
- Kawaguchi, K., et al. (2023). *How Does Information Bottleneck Help Deep Learning?*. ICML 2023.
- Dubois, Y., et al. (2023). *Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression*. arXiv:2305.08013.

**Key Findings:**
- Compression phase depends on neural nonlinearity (tanh vs ReLU)
- Controlling information bottleneck helps control generalization
- Active area of research and debate

**Relevance to Our Work:**
- Provides theoretical lens for understanding LLM compression
- Our work empirically tests information bottleneck in LLMs for QA

---

## 7. Relevant Datasets for Experimentation

### 7.1 Long-Document QA Datasets

**CNN/DailyMail (2015)**
- ~300k news articles
- Cloze-style and abstractive summarization
- Well-established benchmark
- **Use Case**: Test compression on news articles

**NarrativeQA (2017)**
- ~45k QA pairs on books and movie scripts
- Requires reasoning over long contexts (entire books)
- Free-form text answers
- **Use Case**: Test compression on narrative texts

**SQuAD 1.1/2.0 (2016/2018)**
- 107,785 / 151,054 QA pairs on Wikipedia
- Extractive question answering
- Gold standard for QA evaluation
- **Use Case**: Controlled experiments on factual content

### 7.2 Other Relevant Datasets

**NewsQA**
- QA on CNN articles
- 100k+ QA pairs

**MEDIQA-Answer Summarization**
- Question-driven medical summarization
- 156 health questions with summaries

**LongBench (2024)**
- Suite of tasks for long-context evaluation
- Used in LLMLingua evaluation

---

## 8. Novel Aspects of Our Research

### 8.1 What Makes Our Approach Unique?

| Aspect | Prior Work | Our Approach |
|--------|------------|--------------|
| **Compression Goal** | Efficiency, speed, cost reduction | Test limits of semantic information retention |
| **Compression Format** | Human-readable or learned embeddings | Non-human-readable, arbitrary encodings allowed |
| **Model Relationship** | Different models for compression & use | **Same model** creates and interprets |
| **Evaluation** | Reconstruction, perplexity, human judgment | **Downstream QA performance** |
| **Training Required** | Often requires fine-tuning | **Zero-shot** via prompting |
| **Character Set** | Standard Unicode/ASCII | **Arbitrary symbols** encouraged |
| **Serialization** | Often non-serializable (embeddings) | Text-based, can be saved and reloaded |

### 8.2 Key Research Questions

1. **Self-Compression Effectiveness**: Can LLMs design better compression schemes for themselves than for humans?

2. **Information Retention Patterns**: What types of information (facts, entities, relationships, concepts) are preserved vs. lost?

3. **Model-Specific Encodings**: Do models develop unique compression strategies? Are they model-specific?

4. **Comparison to Baselines**: How does this compare to:
   - Human-readable summaries
   - Random sampling
   - Token-based compression (LLMLingua)
   - No compression

5. **Compression Ratio vs. QA Performance**: What is the tradeoff curve?

---

## 9. Gap Analysis: What's Missing in the Literature?

### 9.1 Self-Generated, Non-Human-Readable Compressions

**Finding**: While there are methods that use learned embeddings (Gisting, AutoCompressor, ICAE), none explore letting the model use **arbitrary text-based encodings** that it designs for itself.

**Our Contribution**: We explicitly allow and encourage non-human-readable text compression with arbitrary symbols.

### 9.2 Same-Model Compression and Interpretation

**Finding**: Most compression methods either:
- Use a separate compression model (LLMLingua uses BERT-like models)
- Require fine-tuning (AutoCompressor, Gisting)
- Focus on efficiency rather than testing same-model interpretability

**Our Contribution**: Zero-shot, same-model compression and interpretation to test the model's ability to create self-consistent encodings.

### 9.3 QA-Based Evaluation of Compression

**Finding**: QAEval and similar metrics use QA to **evaluate summarization**, but no prior work uses QA performance as the **primary metric** for evaluating compression schemes where the model designs its own encoding.

**Our Contribution**: Systematic evaluation of self-generated compressions via downstream QA.

### 9.4 Explicit Non-Readability Instruction

**Finding**: Most compression work assumes output should be interpretable by humans or at least use standard vocabulary.

**Our Contribution**: The specific prompt we're testing explicitly states:
> "It does not need to be human readable. You do not need to use a common character set."

This is **novel** and may unlock compression strategies unavailable when constrained to human readability.

---

## 10. Methodology Inspiration from Prior Work

### 10.1 From QAEval: Question Generation and Evaluation

**Adopt:**
- Generate questions automatically from original text
- Use multiple question types (factual, inferential, entity-based)
- Evaluate using F1, Exact Match, and semantic similarity
- Compare to reference answers

### 10.2 From LLMLingua: Compression Ratio Analysis

**Adopt:**
- Systematic compression ratio variation (2x, 5x, 10x, etc.)
- Task-agnostic evaluation
- Analysis of what information is retained

### 10.3 From Semantic Compression (Gilbert et al.): Metrics

**Adapt:**
- Instead of Semantic Reconstruction Effectiveness, use **QA F1/EM**
- Track both exact information and semantic similarity
- Compare compressed vs. full-context QA

### 10.4 From AutoCompressor: Multi-Segment Processing

**Adapt:**
- For very long documents, can compress in segments
- Test whether segmented compression affects QA performance

---

## 11. Experimental Design Implications

Based on this literature review, our experimental design should include:

### 11.1 Baselines (Informed by Prior Work)

1. **Full-Context QA**: Answer with original document (upper bound)
2. **Human-Readable Summary**: Standard summarization prompt
3. **LLMLingua-style Compression**: Token-selection compression
4. **Random Sampling**: Random 1500 tokens from document
5. **No Context**: Answer without any context (lower bound)

### 11.2 Evaluation Metrics (Informed by QAEval)

1. **Exact Match (EM)**
2. **F1 Score** (token overlap)
3. **Semantic Similarity** (BERTScore, embedding cosine similarity)
4. **Question Type Analysis** (factual, inferential, entity-based)
5. **Information Type Analysis** (what's preserved vs. lost)

### 11.3 Document Types (Informed by Dataset Literature)

1. **News Articles** (CNN/DailyMail) - Inverted pyramid structure
2. **Narrative Text** (stories) - Sequential dependencies
3. **Wikipedia Articles** - Factual, structured information
4. **Technical Documents** - Specialized terminology

### 11.4 Compression Variations (Informed by Prompt Compression Work)

Test variations of the compression prompt:
1. **Original prompt** (as specified)
2. **Different token limits** (500, 1000, 1500, 2000)
3. **Different instructions** (focus on facts vs. concepts)
4. **Multi-round compression** (compress the compression)

---

## 12. References

### Semantic Compression
1. Gilbert et al. (2023). Semantic Compression With Large Language Models. arXiv:2304.12512
2. Chevalier et al. (2024). Extending Context Window of Large Language Models via Semantic Compression. ACL 2024 Findings

### Prompt Compression
3. Jiang et al. (2023). LLMLingua: Compressing Prompts for Accelerated Inference. arXiv:2310.05736
4. Pan et al. (2024). LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. arXiv:2403.12968
5. Mu et al. (2023). Learning to Compress Prompts with Gist Tokens. arXiv:2304.08467
6. Chevalier et al. (2023). Adapting Language Models to Compress Contexts. arXiv:2305.14788
7. Ge et al. (2023). In-context Autoencoder for Context Compression in a Large Language Model. arXiv:2307.06945

### Context Compression for QA
8. Chen et al. (2024). CompAct: Compressing Retrieved Documents Actively for Question Answering. arXiv:2407.09014
9. Yang et al. (2025). CompLLM: Compression for Long Context Q&A. arXiv:2509.19228
10. Li et al. (2024). Recurrent Context Compression: Efficiently Expanding the Context Window of LLM. OpenReview

### QA-Based Evaluation
11. Deutsch et al. (2021). Towards Question-Answering as an Automatic Metric for Evaluating the Content Quality of a Summary. TACL
12. Eyal et al. (2019). Question Answering as an Automatic Evaluation Metric for News Article Summarization. NAACL 2019
13. Chen et al. (2024). SummEQuAL: Summarization Evaluation via Question Answering using Large Language Models. ACL 2024

### Lossless Compression
14. Delétang et al. (2024). Language Modeling Is Compression. ICLR 2024. arXiv:2309.10668
15. Valmeekam et al. (2023). LLMZip: Lossless Text Compression using Large Language Models. arXiv:2306.04050
16. Chen et al. (2024). FineZip: Pushing the Limits of Large Language Models for Practical Lossless Text Compression. arXiv:2409.17141

### Information Bottleneck
17. Tishby & Zaslavsky (2015). Deep Learning and the Information Bottleneck Principle. arXiv:1503.02406
18. Kawaguchi et al. (2023). How Does Information Bottleneck Help Deep Learning?. ICML 2023

### Datasets
19. Hermann et al. (2015). Teaching Machines to Read and Comprehend. CNN/DailyMail dataset
20. Kociský et al. (2017). The NarrativeQA Reading Comprehension Challenge
21. Rajpurkar et al. (2016/2018). SQuAD: 100,000+ Questions for Machine Comprehension of Text

---

## 13. Conclusion

**Our research occupies a novel position in the landscape of LLM compression research.**

While extensive work exists on:
- Making LLMs more efficient (prompt compression, context compression)
- Using QA to evaluate summarization quality
- Theoretical frameworks for compression (information bottleneck)

**No prior work combines:**
1. Zero-shot, self-generated compression
2. Explicit permission for non-human-readable encodings
3. Same-model compression and interpretation
4. Systematic QA-based evaluation of compression quality
5. Focus on testing semantic information retention limits

This positions our research as a fundamental study of **LLM self-compression capabilities** rather than an engineering optimization for efficiency.

### Next Steps

With this literature review complete, we can now:
1. Define precise experimental protocols informed by best practices
2. Identify the most relevant baselines and metrics
3. Select appropriate datasets
4. Design the implementation to answer our novel research questions
