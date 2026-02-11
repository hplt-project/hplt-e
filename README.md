# HPLT-e: Comprehensive Multilingual LLM Evaluation

HPLT-e is a framework for comprehensive multilingual and multi-prompt *k*-shot evaluation across 124 tasks in nine typologically diverse languages: Catalan, Spanish, Basque, Galician, French, Norwegian, Ukrainian, Czech, and Finnish.

## 🚀 Updates


* **`19.11.2025`**: We update HPLT-e and release our results of comparing HPLT 3.0, HPLT 2.0, FineWeb 2.1.0, and MADLAD-400 1.0.
* **`02.11.2025`**: A [pre-print](https://arxiv.org/abs/2511.01066) is available, summarizing HPLT-e design principles and early results. 
* **`08.10.2025`**: We make the first release of HPLT-e in connection with the HPLT 3.0 monolingual datasets.


## 📑 Contents
- [🗺️ Overview](#️-overview)
- [🌐 Multilingual Evaluation Suite](#-multilingual-evaluation-suite)
- [🧪 Multilingual Evaluation Recipe](#-multilingual-evaluation-recipe)
- [⚙️ Installation and Usage](#️-installation-and-usage)
- [🧾 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)


## 🗺️ Overview

HPLT-e combines existing monolingual benchmarks for Catalan (CatalanBench), Spanish (SpanishBench), Basque (BasqueBench), Galician (GalicianBench), French (FrenchBench), Norwegian (NorEval), Finnish (FinBench v2), and Czech (BenCzechMark). In addition, we create a multi-task benchmark for Ukrainian (UkrainianBench) and extend single-prompt benchmarks to the multi-prompt scenario (French, Catalan, Spanish, Basque, Galician, and Ukrainian). HPLT-E covers a diverse set of 124 natural language understanding and generation tasks, each supporting 3-7 human-written prompts. Our main evaluation principles include:

* **Diversity**: broader representation of lesser-resourced languages in context of pretraining corpora comparison.
* **Data quality**: use of human-curated datasets to ensure reliable evaluation.
* **Robust evaluation**: evaluation across 500+ prompts written by native speakers to account for prompt sensitivity.
* **Reproducibility**: full integration of HPLT-E into [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) for user-friendly standardized evaluation.


## 🌐 Multilingual evaluation suite

HPLT-e covers different task categories in all languages: entailment, causal reasoning, mathematical reasoning, commonsense reasoning, language knowledge, language-specific & world knowledge, paraphrase detection, reading comprehension, sentiment analysis, toxicity detection, machine translation, and truthfulness. The supported tasks for each language are summarized below.

<details>
<summary><b>Catalan</b></summary>

* Benchmark: CatalanBench
* Paper: [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage: N/A
* Language code: `cat_Latn`
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench)
* HPLT-e multi-prompt implementation: [cat_Latn](./cat_Latn/)


|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|	ARC-ca	| `arc_ca_challenge_p[0-2]`	|	Multiple-choice QA |	Language-specific & world knowledge |
|	ARC-ca	| `arc_ca_easy_p[0-2]`		|	Multiple-choice QA |	Language-specific & world knowledge |
|	Belebele| 	`catbelebele_p[0-2]`	|Multiple-choice QA |	Reading comprehension |
|	CatalanQA|	`catalanqa_p[0-2]`			 |Generative QA	| Language-specific & world knowledge|
|	CatCoLA|	`catcola_p[0-2]`	|		 Text classification	|Language knowledge|
|	COPA-ca	|`copa_ca_p[0-2]`	|	Text cassification	|Commonsense reasoning|
|	CoQCat	|`coqcat_p[0-2]`	|	 Generative QA	|Reading comprehension|
|	MGSM-cat|	`mgsm_direct_ca_p[0-2]`	|	Generative QA	|Mathematical reasoning|
|	OpenBookQA-cat	|`openbookqa_ca_p[0-2]`	|	Multiple-choice QA |	Language-specific & world knowledge|
|	Parafraseja	|`parafraseja_p[0-2]`	|	Text classification	|Paraphrase detection|
|	PAWS-ca	|`paws_ca_p[0-2]`	|	 Text classification	|Paraphrase detection|
|	PIQA-ca|	`piqa_ca_p[0-2]`	|		Multiple-choice QA |	Commonsense reasoning|
|	SIQA-ca	|`siqa_ca_p[0-2]`	|		Multiple-choice QA |	Commonsense reasoning|
|	TE-ca	|`teca_p[0-2]`	|	Text classification	|Entailment|
|	VeritasQA-cat Generation	|`veritasqa_ca_gen_p[0-2]`	|	Generative QA|	Truthfulness|
|	VeritasQA-cat Multiple-choice	|`veritasqa_ca_mc1_p[0-2]`	|	Multiple-choice QA |	Truthfulness|
|	VeritasQA-cat Multiple-choice	|`veritasqa_ca_mc2_p[0-2]`	|	Multiple-choice QA |	Truthfulness|
|	WNLI	|`wnli_ca_p[0-2]`	|	Text classification	|Entailment|
|	XNLI	|`xnli_ca_p[0-2]`	|	Text classification	|Entailment|
|	XQuAD	|`xquad_ca_p[0-2]`	|	Generative QA|	Reading comprehension|
|	xStoryCloze	|`xstorycloze_ca_p[0-2]`	|	Multiple-choice QA |	Commonsense reasoning|
|	Cocoteros	|`cocoteros_va_p[0-2]`	|	Text generation	|Commonsense reasoning|
|	FLORES	| `flores_en-ca_p[0-2]`	|	Sequence-to-sequence generation|	Machine translation|

</details>

<details>
<summary><b>Spanish</b></summary>

* Benchmark: SpanishBench
* Paper: [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage: N/A
* Language code: `spa_Latn`
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/spanish_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/spanish_bench)
* HPLT-e multi-prompt implementation: [spa_Latn](./spa_Latn/)

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|Belebele|	`spabelebele_p[0-2]`	 |Multiple-choice QA	|Reading comprehension|
|COPA|	`copa_es_p[0-2]`		|Text cassification	|Commonsense reasoning|
|ESCoLA	|`escola_p[0-2]`		|Text cassification	|Language knowledge|
|MGSM-es|	`mgsm_direct_es_p[0-2]`	|	Generative QA|	Mathematical reasoning|
|OpenBookQA-es	|`openbookqa_es_p[0-2]`	|	Multiple-choice QA|	Language-specific & world knowledge|
|PAWS-es|	`paws_es_p[0-2]`	|	Text cassification|	Paraphrase detection|
|VeritasQA-es Generation	|`veritasqa_es_gen_p[0-2]`	|	Generative QA|	Truthfulness|
|VeritasQA-es Multiple-choice	|`veritasqa_es_mc1_p[0-2]`	|	Multiple-choice QA	|Truthfulness|
|VeritasQA-es Multiple-choice	|`veritasqa_es_mc2_p[0-2]`	|		Multiple-choice QA|	Truthfulness|
|XNLI	| `xnli_es_p[0-2]`	|	Text cassification	| Entailment|
|XQuAD	| `xquad_es_p[0-2]`|		Generative QA	|Reading comprehension|
|xStoryCloze	|`xstorycloze_es_p[0-2]`	|	Multiple-choice QA|	Commonsense reasoning|
|Cocoteros|	`cocoteros_es_p[0-2]`	|		Text generation | 	Commonsense reasoning|
|FLORES	|`flores_en-es_p[0-2]`	| Sequence-to-sequence generation|	Machine translation|
|INCLUDE|	`include_spanish_p[0-2]`		|	Multiple-choice QA	|Language-specific & world knowledge|
|Global-MMLU|	`global_mmlu_spanish_p[0-2]`		|	Multiple-choice QA	|Language-specific & world knowledge|

</details>

<details>
<summary><b>Galician</b></summary>

* Benchmark: GalicianBench
* Paper: [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage: N/A
* Language code: `glg_Latn`
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/galician_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/galician_bench)
* HPLT-e multi-prompt implementation: [glg_Latn](./glg_Latn/)


|Name  |LM Evaluation Harness   | Task type  | Task category |
|:---|:---|:---|:---|
|	Belebele |	`glgbelebele_p[0-2]`	|	 	Multiple-choice QA	|Reading comprehension|
|	MGSM-gl	|`mgsm_direct_gl_p[0-2]`	|		Generative QA	|Mathematical reasoning|
| GalCoLA | `galcola_p[0-2]`      | Text classification| Language knowledge|
|	OpenBookQA-gl	|`openbookqa_gl_p[0-2]`	|	Multiple-choice QA |	Language-specific & world knowledge|
|	Parafrases-gl	|`parafrases_gl_p[0-2]`	|	Text classification	|Paraphrase detection|
|	PAWS-gl|	`paws_gl_p[0-2]`	|Text classification|	Paraphrase detection|
|	FLORES	|`flores_en-glg_p[0-2]`	|	Sequence-to-sequence generation|	Machine translation|
|	VeritasQA-gl Generation	|`veritasqa_gl_gen_p[0-2]`	|	Generative QA|	Truthfulness|
|	VeritasQA-gl Multiple-choice	|`veritasqa_gl_mc1_p[0-2]`	|	Multiple-choice QA |	Truthfulness|
|	VeritasQA-gl Multiple-choice	|`veritasqa_gl_mc2_p[0-2]`	|	Multiple-choice QA |	Truthfulness|

</details>

<details>
<summary><b>Basque</b></summary>

* Benchmark: BasqueBench
* Paper: [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage: N/A
* Language code: `eus_Latn`
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/basque_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/basque_bench)
* HPLT-e multi-prompt implementation: [eus_Latn](./eus_Latn/)


|Name  |LM Evaluation Harness   | Task type  | Task category |
|:---|:---|:---|:---|
|	Belebele |	`eusbelebele_p[0-2]`	|	 	Multiple-choice QA	|Reading comprehension|
|	EusExams	|`eus_exams_eu_p[0-2]`	|		Multiple-choice QA	|Language-specific & world knowledge|
|	EusProfficiency	|`eus_proficiency_p[0-2]`	|	 Multiple-choice QA	|Language-specific & world knowledge|
|	EusReading|	`eus_reading_p[0-2]`	|	 Multiple-choice QA|	Reading comprehension|
|	EusTrivia	|`eus_trivia_p[0-2]`	|	 Multiple-choice QA|	Language-specific & world knowledge|
|	MGSM-eu	|`mgsm_direct_eu_p[0-2]`	|		Generative QA	|Mathematical reasoning|
|	PIQA-eu	|`piqa_eu_p[0-2]`	|	Multiple-choice QA|	Commonsense reasoning|
|	WNLI	|`wnli_eu_p[0-2]`	|	Text classification|	Entailment|
|	XCOPA	|`xcopa_eu_p[0-2]`	|	Text cassification	|Commonsense reasoning|
|	XNLI	|`xnli_eu_native_p[0-2]`	|	Text classification|	Entailment|
|	xStoryCloze|	`xstorycloze_eu_p[0-2]`	|		Multiple-choice QA|	Commonsense reasoning|
|	PAWS-eu|	`paws_eu_p[0-2]`	|Text classification|	Paraphrase detection|
|	ARC-eu|`arc_eu_easy_p[0-2]`	|	 	Multiple-choice QA|	Language-specific & world knowledge|
|	ARC-eu|	`arc_eu_challenge_p[0-2]`	|		Multiple-choice QA|	Language-specific & world knowledge|
|	FLORES	|`flores_en-eu_p[0-2]`	|	Sequence-to-sequence generation|	Machine translation|
|	INCLUDE	|`include_basque_p[0-2]`	|		Multiple-choice QA|	Language-specific & world knowledge|

</details>

<details>
<summary><b>Norwegian</b></summary>

* Benchmark: NorEval
* Paper: [aclanthology.org/2025.findings-acl.181](https://aclanthology.org/2025.findings-acl.181/)
* Homepage: [github.com/ltgoslo/noreval](https://github.com/ltgoslo/noreval/tree/main)
* Language code: `nor_Latn` (Bokmål and Nynorsk)
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval)
* multi-prompt implementation: N/A


|Name  |LM Evaluation Harness (Bokmål) | LM Evaluation Harness (Nynorsk)  |Task type  | Task category |
|:---|:---|:---|:---|:---|
|[NoReC Sentence](https://huggingface.co/datasets/ltg/norec_sentence) |```norec_sentence_p[0-4]```  | ❌ |Text classification| Sentiment analysis |
|[NoReC Document](https://huggingface.co/datasets/ltg/norec_document) |```norec_document_p[0-4]```  | ❌ |Text classification| Sentiment analysis |
|[NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) |```noridiom_nob_p[0-4]```  | ```noridiom_nno_p[0-4]```  | Sentence completion| Language knowledge  |
|[Belebele](https://huggingface.co/datasets/facebook/belebele) |```norbelebele_p[0-4]```| ❌|Multiple-choice QA| Reading comprehension |
|[NRK-Quiz-QA](https://huggingface.co/datasets/ltg/nrk_quiz_qa) |```nrk_quiz_qa_nob_p[0-4]```| ```nrk_quiz_qa_nno_p[0-4]```| Multiple-choice QA| Language-specific & world knowledge |
|[NorOpenBookQA](https://huggingface.co/datasets/ltg/noropenbookqa) |```noropenbookqa_nob_p[0-4]```| ```noropenbookqa_nno_p[0-4]``` |Multiple-choice QA| Language-specific & world knowledge |
|[NorCommonsenseQA](https://huggingface.co/datasets/ltg/norcommonsenseqa) |```norcommonsenseqa_nob_p[0-4]```| ```norcommonsenseqa_nno_p[0-4]``` |Multiple-choice QA|Commonsense reasoning  |
|[NorTruthfulQA Multiple choice](https://huggingface.co/datasets/ltg/nortruthfulqa_mc) |```nortruthfulqa_mc_nob_p[0-4]```| ```nortruthfulqa_mc_nno_p[0-4]``` |Multiple-choice QA |Truthfulness |
|[NorQuAD](https://huggingface.co/datasets/ltg/norquad) |```norquad_p[0-4]```| ❌  | Generative QA |Reading comprehension |
|[NorTruthfulQA Generation](https://huggingface.co/datasets/ltg/nortruthfulqa_gen) |```nortruthfulqa_gen_nob_p[0-4]```| ```nortruthfulqa_gen_nno_p[0-4]``` |  Generative QA|Truthfulness |
|[Tatoeba (English → Bokmål/Nynorsk)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | ```tatoeba_eng_nob_p[0-4]```| ```tatoeba_eng_nno_p[0-4]```  |Sequence-to-sequence generation|Machine translation |

</details>


<details>
<summary><b>Ukrainian</b></summary>

* Benchmark: UkrainianBench
* Paper: [arxiv.org/abs/2511.01066](https://arxiv.org/abs/2511.01066)
* Homepage: [github.com/hplt-project/](https://github.com/hplt-project/)
* Language code: `ukr_Cyrl`
* Original LM Evaluation Harness implementation: N/A 
* HPLT-e multi-prompt implementation: [ukr_Cyrl](./ukr_Cyrl/)

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|[Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU)| `global_mmlu_ukrainian_p[0-2]` | Multiple-choice QA | Language-specific & world knowledge |
|[ZNO](https://huggingface.co/datasets/osyvokon/zno)| `zno_p[0-2]` |  Multiple-choice QA | Language-specific & world knowledge |
|[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)| `include_ukrainian_p[0-2]` |  Multiple-choice QA | Language-specific & world knowledge |
|[TextDetox](https://huggingface.co/datasets/ukr-detect/ukr-toxicity-dataset)| `textdetox_ukr_p[0-2]` | Text classification | Toxicity detection | 
|[UA-SQuAD](https://huggingface.co/datasets/HPLT/ua-squad) | `ua_squad_p[0-2]` |  Generative QA | Reading comprehension |
|[Belebele](https://huggingface.co/datasets/facebook/belebele) | `ukrbelebele_p[0-2]` |  Multiple-choice QA | Reading comprehension |
|[WMT24PP](https://huggingface.co/datasets/google/wmt24pp/) | `wmt24pp_en-uk_p[0-2]` |  Sequence-to-sequence generation	| Machine translation|

</details>


<details>
<summary><b>Czech</b></summary>

* Benchmark: BenCzechMark
* Paper: [direct.mit.edu/tacl/article/doi/10.1162/TACL.a.32/132962](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.32/132962/BenCzechMark-A-Czech-Centric-Multitask-and)
* Homepage: [github.com/DCGM/lm-evaluation-harness](https://github.com/DCGM/lm-evaluation-harness)
* Language code: `ces_Latn`
* Original LM Evaluation Harness implementation: [github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark](https://github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark)
* HPLT-e multi-prompt implementation: [ces_Latn](./ces_Latn/)

**NB**: we update BenCzechmark to enable support for latest LM Evaluation Harness versions and create new prompts for Global-MMLU.

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
 |	Belebele	|`cesbelebele_p[0-4]`	 |	 Multiple-choice QA  | 	Reading comprehension |
 |	Global-MMLU|	`global_mmlu_czech_p[0-4]`	 |		Multiple-choice QA  | 	Language-specific & world knowledge |
 |	SQAD3.2	|`cs_sqad32_p[0-4]`	 |	Generative QA	| Reading comprehension| 
 |	Umimeto 	|`umimeto_p[0-4]`	 |	Multiple-choice QA  | 	Language-specific & world knowledge| 
 |	CERMAT OPEN	|`cermat_czech_open_p[0-4]`	 |	Generative QA |	Language knowledge|
 |	CERMAT TF	|`cermat_czech_tf_p[0-4]`	 |	Multiple-choice QA  | 	Language knowledge|
 |	CERMAT MC	|`cermat_czech_mc_p[0-4]`	 |	Multiple-choice QA  | 	Language knowledge|
 |	Klokan QA |	`klokan_qa_p[0-4]`	 |	Multiple-choice QA  | 	Mathematical reasoning|
 |	CERMAT (Math) MC |	`cermat_czmath_mc_p[0-4]`	 |	Multiple-choice QA  | 	Mathematical reasoning|
  |	CERMAT (Math) OPEN|	`cermat_czmath_open_p[0-4]`	 |	Generative QA  | 	Mathematical reasoning|
 |	CTKFacts 	| `ctkfacts_nli_p[0-4]`	 |		Text classification | 	Entailment|
 |	Subjectivity 	|`ces_subjectivity_p[0-4]`	 |	Text classification | 	Sentiment analysis|
 |	CzechSentiment - Mall	|`sentiment_mall_p[0-4]`	 |		Text classification | 	Sentiment analysis|
 |	CzechSentiment - CSFD |	`sentiment_csfd_p[0-4]`	 |		Text classification | 	Sentiment analysis|
 |	CzechSentiment - FB |	`sentiment_fb_p[0-4]`	 |	Text classification | 	Sentiment analysis|

</details>

<details>
<summary><b>French</b></summary>

* Benchmark: FrenchBench
* Paper: [arxiv.org/abs/2402.00786](https://arxiv.org/abs/2402.00786)
* Homepage: [huggingface.co/croissantllm](https://huggingface.co/croissantllm)
* Language code: `fra_Latn`
* Original LM Evaluation Harness implementation: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench)
* HPLT-e multi-prompt implementation: [fra_Latn](./fra_Latn/)

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|	FQuaD	|`fquad_p[0-2]`	|					Generative QA|	Reading comprehension|
|	French Language Test: Grammar	|`french_bench_grammar_p[0-2]`	|					Multiple-choice QA|	Language knowledge|
|	French Language Test: Vocabulary|	`french_bench_vocabulary_p[0-2]`	|					Multiple-choice QA|	Language knowledge|
|	French Language Test: Reading	|`french_bench_reading_p[0-2]`	|					Multiple-choice QA|	Reading comprehension|
|	Belebele	|`frabelebele_p[0-2]`	|					Multiple-choice QA|	Reading comprehension|
|	French NLI|	`topic_based_nli_p[0-2]`	|					Text classification|	Entailment|
|	XNLI	|`french_xnli_p[0-2]`	|					Text classification	|Entailment|
|	INCLUDE	|`include_french_p[0-2]`	|	 Multiple-choice QA|	Language-specific & world knowledge|
|	Global-MMLU	| `global_mmlu_french_p[0-2]`	|		Multiple-choice QA|	Language-specific & world knowledge|

</details>

<details>
<summary><b>Finnish</b></summary>

* Benchmark: FinBench v2
* Paper: TBA
* Homepage: N/A
* Language code: `fin_Latn`
* Original LM Evaluation Harness implementation: [github.com/LumiOpen/lm-evaluation-harness/tree/finbench_v2/lm_eval/tasks/finbench_v2](https://github.com/LumiOpen/lm-evaluation-harness/tree/finbench_v2/lm_eval/tasks/finbench_v2)
* HPLT-e multi-prompt implementation: N/A

| Name| Formulation | LM Evaluation Harness| Task type | Task category| FinBench v2 dataset version |
|:--------------------|-------------|:----|:----------------|:----|:------------------|
| [ARC-challenge-fi](https://huggingface.co/datasets/silogen/ARC-C-fi-HT) |Multiple-choice | `arc_challenge_fi_mcf_fbv2_p[0-4]` | Multiple-choice QA | Language-scpecific & world knowledge | [finbenchv2-arc-c-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-arc-c-fi-ht)  | 
|     |Close| `arc_challenge_fi_cf_fbv2_p[0-4]`  ||  |      |    |
| [Belebele](https://huggingface.co/datasets/facebook/belebele)   |Multiple-choice | `belebele_fin_Latn_mcf_fbv2_p[0-4]`| Multiple-choice QA | Reading comprehension | [finbenchv2-belebele-fi-og](https://huggingface.co/datasets/TurkuNLP/finbenchv2-belebele-fi-og)  |
|     |Close| `belebele_fin_Latn_cf_fbv2_p[0-4]` ||  |      |    |
| [GoldenSwag](https://huggingface.co/datasets/PleIAs/GoldenSwag) |Multiple-choice | `goldenswag_ht_fi_mcf_fbv2_p[0-4]`  | Sentence completion  | Commonsense reasoning | [finbenchv2-goldenswag-fi-ht](https://huggingface.co/datasets/TurkuNLP/finbenchv2-goldenswag-fi-ht)      |
|     |Close| `goldenswag_ht_fi_cf_fbv2_p[0-4]`  ||  |      |    |
| [FIN-Bench](https://github.com/TurkuNLP/FIN-bench)     |Multiple-choice | `finbench_analogies_mcf_fbv2_p[0-4]`   | Multiple choice QA | Relational reasoning  | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_analogies_cf_fbv2_p[0-4]`||  |      |    |
|     |Multiple-choice | `finbench_emotions_mcf_fbv2_p[0-4]` | Text classification | Sentiment analysis    | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_emotions_cf_fbv2_p[0-4]` ||  |      |    |
|     |Multiple-choice | `finbench_empirical_judgments_mcf_fbv2_p[0-4]` | Text classification | Causal reasoning      | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_empirical_judgments_cf_fbv2_p[0-4]` ||  |      |    |
|     |Multiple-choice | `finbench_general_knowledge_mcf_fbv2_p[0-4]` | Multiple choice QA | Language-scpecific & world knowledge | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_general_knowledge_cf_fbv2_p[0-4]` ||  |      |    |
|     |Multiple-choice | `finbench_hhh_alignment_mcf_fbv2_p[0-4]`     | Multiple choice QA | Alignment and safety  | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_hhh_alignment_cf_fbv2_p[0-4]`     ||  |      |    |
|     |Multiple-choice | `finbench_paraphrase_mcf_fbv2_p[0-4]`  | Text classification | Paraphrase detection     | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_paraphrase_cf_fbv2_p[0-4]`  ||  |      |    |
|     |Multiple-choice | `finbench_similarities_abstraction_mcf_fbv2_p[0-4]`  | Multiple choice QA | Commonsense reasoning | [FIN-bench](https://huggingface.co/datasets/TurkuNLP/FIN-bench)  |
|     |Close| `finbench_similarities_abstraction_cf_fbv2_p[0-4]`  ||  |      |    |


</details>


## 🧪 Multilingual Evaluation Recipe
We provide results of our ablation studies evaluating different corpora and sampling strategies across multiple languages:

* [**⚖️ HPLT Pre-3.0 Comparison**](./results/2505-deduplication/README.md) (May 2025): Comparison of data deduplication strategies on a pre-release version of HPLT 3.0, across nine selected languages.
* [**📚 Corpora Comparison**](./results/2508-datasets/README.md) (August 2025): Contrastive dataset evaluation of HPLT 2.0, HPLT 3.0, FineWeb 2.1.0, and MADLAD-400 1.0, on nine selected languages.
* [**🧰 Web Document Scorer (WDS) Comparison**](./results/2508-wds/README.md) (August 2025): Analysis of HPLT 3.0 corpora sampled using different WDS thresholds, focusing on Spanish and French.


Our multilingual evaluation recipe consists of three main components:
* 🧩 **Pretraining**: Pretraining ablation models on various corpora configurations for the target languages.
* 🎯 **Task selection**: Selecting tasks that provide reliable pretraining evaluation signal based on the maximum performance across prompts.
* 📊 **Performance aggregation**: Aggregating performance on the selected tasks across languages.


<details>
<summary><b>🧩 Pretraining</b></summary>

Each evaluation series involves pretraining individual 2.15B-parameter models for every language, following a fixed pretraining setup. All models follow the Llama architecture with 24 layers, 32 attention heads, and a sequence length of 2048. The tokenizer is Gemma-3 with the vocabulary size of 262K tokens. For lower-resource languages with less than 30B/100B tokens of available data, datasets are uniformly upsampled (repeated) following [Muennighoff et al. (2023)](https://openreview.net/forum?id=j5BuTrEj35). Pretraining is run using the Megatron-LM framework on the LUMI supercomputer, employing 16 AMD MI250x nodes and totaling approximately 1k GPU hours.

</details>

<details>
<summary><b>🎯 Task selection</b></summary>

We use the standard task-specific metrics and report the maximum score across the prompts as the main performance aggregation method. We extend [the FineWeb 2.1.0 evaluation design](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fine-tasks) to examine the signal HPLT-e tasks provide based on the criteria and statistics summarized below.

- **Monotonicity**: performance should improve as pretraining progresses, even if the improvement differs across pretraining corpora. Tasks with fluctuating scores promote limited reliability.
- **Stable pretraining**: relative variability of performance across checkpoints should be low, reflecting smooth pretraining dynamics.
- **Ranking consistency**: relative ranking of models should remain consistent across consecutive pretraining intervals.
- **Prompt sensitivity**: performance should be consistent across various prompt formulations.
- **Prompt-switch rate**: frequent switches in best-performing prompt further reflects low evaluation reliability due to potential prompt lottery.
- **Signal-to-Noise ratio**: differences in task performance should primarily reflect differences in corpora quality, not random variation due to prompt choice.
- **Non-randomness**: final checkpoints should achieve performance above a random guessing baseline. Tasks where all models perform near random provide low discriminative power.

Specific evaluation criteria requirements are detailed on the corresponding evaluation page.

</details>

<details>
<summary><b>📊 Performance aggregation</b></summary>
We select tasks that provide the pretraining evaluation signal and aggregate the performance using a combination of several approaches.

#### 🔤 Language score

We compute **language scores** across the selected tasks as follows:

1. **Rescaling**: Normalize performance scores relative to a random baseline using min–max normalization.
2. **Category averaging**: Compute the average of normalized scores within each task category.
3. **Language score**: Derive the final language-level score as the mean of the category averages.


#### 🌍 "Multilingual" score

To compute the **multilingual score**, we utilize several approaches:

1. **Average normalized score**: We average min-max normalized language scores.
2. **Average rank**: We rank the final checkpoints' language scores across all corpora configurations and average their ranks.
3. **Borda count**: First, we rank the final checkpoints for each language; second, we apply the Borda count on the language-wise rankings to compute the final ranking. We utilize the [Vote'n'Rank](https://aclanthology.org/2023.eacl-main.48/) framework.

</details>

### 😎 HPLT-e Tasks

Based on our large-scale evaluations, we find that the set of selected tasks for each language can slightly differ depending on the number of corpora and checkpoints included in the comparison. Although we encourage users to perform task selection on their own data using our codebase, we also release a set of **HPLT-e tasks** for languages that less represented in multilingual evaluations, derived from [our 100BT model evaluation results](results/2508-datasets/README.md).

<details>
<summary><b>😎 HPLT-e tasks</b></summary>

|Language |Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|:---|
|Spanish|COPA|	`copa_es_p[0-2]`		|Text cassification	|Commonsense reasoning|
|Spanish|OpenBookQA-es	|`openbookqa_es_p[0-2]`	|	Multiple-choice QA|	Language-specific & world knowledge|
|Spanish|XNLI	| `xnli_es_p[0-2]`	|	Text cassification	| Entailment|
|Spanish|xStoryCloze	|`xstorycloze_es_p[0-2]`	|	Multiple-choice QA|	Commonsense reasoning|
|Catalan|	ARC-ca	| `arc_ca_easy_p[0-2]`		|	Multiple-choice QA |	Language-specific & world knowledge |
|Catalan|	COPA-ca	|`copa_ca_p[0-2]`	|	Text cassification	|Commonsense reasoning|
|Catalan|	CoQCat	|`coqcat_p[0-2]`	|	 Generative QA	|Reading comprehension|
|Catalan|	PIQA-ca|	`piqa_ca_p[0-2]`	|		Multiple-choice QA |	Commonsense reasoning|
|Catalan|	SIQA-ca	|`siqa_ca_p[0-2]`	|		Multiple-choice QA |	Commonsense reasoning|
|Catalan|	TE-ca	|`teca_p[0-2]`	|	Text classification	|Entailment|
|Catalan|	xStoryCloze	|`xstorycloze_ca_p[0-2]`	|	Multiple-choice QA |	Commonsense reasoning|
|Ukrainian| ZNO | `zno_p[0-2]` |  Multiple-choice QA | Language-specific & world knowledge |
|Ukrainian| UA-SQuAD | `ua_squad_p[0-2]` |  Generative QA | Reading comprehension |
|Czech|	SQAD3.2	|`cs_sqad32_p[0-4]`	 |	Generative QA	| Reading comprehension| 
|Finnish| ARC-challenge-fi | `arc_challenge_fi_cf_fbv2_p[0-4]` | Multiple-choice QA | Language-scpecific & world knowledge |
|Finnish| Belebele  |`belebele_fin_Latn_cf_fbv2_p[0-4]`| Multiple-choice QA | Reading comprehension |
|Finnish| GoldenSwag | `goldenswag_ht_fi_cf_fbv2_p[0-4]`  | Sentence completion  | Commonsense reasoning |
|Finnish| FIN-Bench Analogies    | `finbench_analogies_cf_fbv2_p[0-4]`   | Multiple choice QA | Language-scpecific & world knowledge  | 
|Finnish|FIN-bench General Knowledge | `finbench_general_knowledge_cf_fbv2_p[0-4]` | Multiple choice QA | Language-scpecific & world knowledge |
|French|	FQuaD	|`fquad_p[0-2]`	|					Generative QA|	Reading comprehension|
|French|	French Language Test: Vocabulary|	`french_bench_vocabulary_p[0-2]`	|					Multiple-choice QA|	Language knowledge|
|Norwegian|NorIdiom Nynorsk |```noridiom_nno_p[0-4]```  | Sentence completion| Language knowledge  |
|Norwegian|NRK-Quiz-QA Bokmål |```nrk_quiz_qa_nob_p[0-4]```| Multiple-choice QA| Language-specific & world knowledge |
|Norwegian|NRK-Quiz-QA Nynorsk | ```nrk_quiz_qa_nno_p[0-4]```| Multiple-choice QA| Language-specific & world knowledge |
|Norwegian|NorCommonsenseQA Bokmål |```norcommonsenseqa_nob_p[0-4]```|Multiple-choice QA|Commonsense reasoning  |
|Norwegian|NorQuAD|```norquad_p[0-4]```| Generative QA |Reading comprehension |

</details>


## ⚙️ Installation and usage

1. Install LM Evaluation Harness as described [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

2. Clone our HPLT-e GitHub repository.

```bash
git clone https://github.com/hplt-project/hplt-e.git
cd hplt-e
```

3. Get the `finbench_v2` folder from [the FinBench v2 GitHub repository](https://github.com/LumiOpen/lm-evaluation-harness/tree/finbench_v2/lm_eval/tasks/finbench_v2).


<details>
<summary><b> 💻 How to run evaluation?</b></summary>

Detailed guidelines on how to use LM Evaluation Harness can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md). The task names can be found in the **LM Evaluation Harness** column in the language-specific task tables provided above. `_p[i-j]` stands for the corresponding supported prompts.

#### Basic usage

Below is an example of a basic framework usage and must-have arguments. The evaluation requires the usage of the `include_path` argument to ensure our tasks are registered in the framework as these are not part of LM Evaluation Harness yet:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=my_hf_model_name \
  --tasks global_mmlu_ukrainian_p0 \
  --include_path ./
  --output results/ukrainian/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

An example of a slurm script for [the LUMI supercomputer](https://www.lumi-supercomputer.eu) is provided [here](scripts/run.sh).

```bash
sbatch scripts/run.sh <model_name> <task_name>
```

#### Task groups

An alternative approach to run all tasks of interest at once involves creating a task group. LM Evaluation Harness allows to group tasks as described [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#group-configuration). An example for the Ukrainian `global_mmlu_ukrainian_p0` group task can be found [here](ukr_Cyrl/global_mmlu/global_mmlu_ukrainian_p0.yaml).


</details>

<details>
<summary><b> 🗂️ How to select tasks?</b></summary>

Please find the example on how to select tasks and vizualize the results below.

```python
from datasets import load_dataset
from utils import *
from viz import *

dataset = load_dataset("HPLT/corpora-comparison-evals", "spa_Latn", split="results").to_pandas()
dataset_criteria_results, dataset_normalized_results, dataset_filtered_tasks = get_normalized_results(
    dataset,
    score_col="max_score",
    monotonicity_threshold=0.5,
    snr_threshold=3,
    mad_threshold=5,
    cv_threshold=15,
    higher_bound=100,
    thresholds=THRESHOLDS_100BT
)
```

```python
# Viz. example 1: Plotting the results on the selected tasks.
plot_normalized_results(
  dataset_normalized_results,
  tick_style="100BT" # tick_style="30BT" for the 30BT models 
) 
```

```python
# Viz. example 2: Plotting the results on particular task (works for non- and selected tasks). 
plot_results_by_task(
  raw_results=dataset,
  task="spabelebele",
  score_col="max_score", # available: "max_score", "median_score", and "mean_score"
  tick_style="100BT" # tick_style="30BT" for the 30BT models 
)
```


We provide detailed codebase on task selection [here](utils.py), vizualization codebase [here](viz.py), and the recommended thresholds for [30BT](configs/thresholds_30bt.json) and [100BT](configs/thresholds_100bt.json) models.

</details>


## 🧾 Citation

Our [pre-print](https://arxiv.org/abs/2511.01066) is available on arXiv.

```
@article{oepen2025hplt,
  title={HPLT\~{} 3.0: Very Large-Scale Multilingual Resources for LLM and MT. Mono-and Bi-lingual Data, Multilingual Evaluation, and Pre-Trained Models},
  author={Oepen, Stephan and Arefev, Nikolay and Aulamo, Mikko and Ba{\~n}{\'o}n, Marta and Buljan, Maja and Burchell, Laurie and Charpentier, Lucas and Chen, Pinzhen and Fedorova, Mariya and de Gibert, Ona and others},
  journal={arXiv preprint arXiv:2511.01066},
  year={2025}
}
```

## 🙏 Acknowledgements
This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546].

We thank Étienne Simon (UiO), Lucas Georges Gabriel Charpentier (UiO), and Daryna Dementieva (TUM) for their contribution to our prompt collection for French and Ukrainian.
