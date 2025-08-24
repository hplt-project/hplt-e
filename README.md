# HPLT-E: Multilingual and Comprehensive LLM Evaluation

## Contents
- [Overview](#overview)
- [Evaluation suite](#evaluation-suite)
- [Installation and Usage](#installation-and-usage)


## Overview

* Task taxonomy (Google doc): [here](https://docs.google.com/spreadsheets/d/13DjTlr4Ph_QSFvRI2kmIoB0gdEqAGOPBQ_J2SgTg2tQ/edit?usp=sharing)
* Current version of the taxonomy: [here](taxonomy.csv)
* Explorative analysis: [here](explorative.ipynb)

## Evaluation suite


### Catalan

* Benchmark(s): CatalanBench
* Paper(s): [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage(s): N/A
* Language code(s): `cat_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench)


<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|	ARC-ca	| `arc_ca_challenge`	|	Multiple-choice QA |	Language-specific & world knowledge |
|	ARC-ca	| `arc_ca_easy`		|	Multiple-choice QA |	Language-specific & world knowledge |
|	Belebele| 	`belebele_cat_Latn`	|Multiple-choice QA |	Reading comprehension |
|	CatalanQA|	`catalanqa`			 |Generative QA	| Language-specific & world knowledge|
|	CatCoLA|	`catcola`	|		 Text classification	|Language knowledge|
|	COPA-ca	|`copa_ca`	|	Text cassification	|Commonsense reasoning|
|	CoQCat	|`coqcat`	|	 Generative QA	|Reading comprehension|
|	MGSM-cat|	`mgsm_direct_ca`	|	Generative QA	|Mathematical reasoning|
|	OpenBookQA-cat	|`openbookqa_ca`	|	Multiple-choice QA |	Language-specific & world knowledge|
|	Parafraseja	|`parafraseja`	|	Text classification	|Paraphrasing|
|	PAWS-ca	|`paws_ca`	|	 Text classification	|Paraphrasing|
|	PIQA-ca|	`piqa_ca`	|		Multiple-choice QA |	Commonsense reasoning|
|	SIQA-ca	|`siqa_ca`	|		Multiple-choice QA |	Commonsense reasoning|
|	TE-ca	|`TE-ca`	|	Text classification	|Entailment|
|	VeritasQA-cat Generation	|`veritasqa_gen_ca`	|	Generative QA|	Truthfulness|
|	VeritasQA-cat Multiple-choice	|`veritasqa_mc1_ca`	|	Multiple-choice QA |	Truthfulness|
|	VeritasQA-cat Multiple-choice	|`veritasqa_mc2_ca`	|	Multiple-choice QA |	Truthfulness|
|	WNLI	|`wnli_ca`	|	Text classification	|Entailment|
|	XNLI	|`xnli_ca`	|	Text classification	|Entailment|
|	XQuAD	|`xquad_ca`	|	Generative QA|	Reading comprehension|
|	xStoryCloze	|`xstorycloze_ca`	|	Multiple-choice QA |	Commonsense reasoning|
|	Cocoteros	|`cocoteros_va`	|	Text generation	|Commonsense reasoning|
|	FLORES	| `flores_en-ca`	|	Sequence-to-sequence generation|	Machine translation|

</details>

### Spanish

* Benchmark(s): SpanishBench
* Paper(s): [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage(s): N/A
* Language code(s): `spa_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/spanish_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/spanish_bench)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|Belebele|	`belebele_spa_Latn`	 |Multiple-choice QA	|Reading comprehension|
|COPA|	`copa_es`		|Text cassification	|Commonsense reasoning|
|ESCoLA	|`escola`		|Text cassification	|Language knowledge|
|MGSM-es|	`mgsm_direct_es`	|	Generative QA|	Mathematical reasoning|
|OpenBookQA-es	|`openbookqa_es`	|	Multiple-choice QA|	Language-specific & world knowledge|
|PAWS-es|	`paws_es`	|	Text cassification|	Paraphrasing|
|VeritasQA-es Generation	|`veritasqa_gen_es`	|	Generative QA|	Truthfulness|
|VeritasQA-es Multiple-choice	|`veritasqa_mc1_es`	|	Multiple-choice QA	|Truthfulness|
|VeritasQA-es Multiple-choice	|`veritasqa_mc2_es`	|		Multiple-choice QA|	Truthfulness|
|WNLI	| `wnli_es`	|	Text cassification	|Entailment|
|XNLI	| `xnli_es`	|	Text cassification	| Entailment|
|XQuAD	| `xquad_es`|		Generative QA	|Reading comprehension|
|xStoryCloze	|`xstorycloze_es`	|	Multiple-choice QA|	Commonsense reasoning|
|Cocoteros|	`cocoteros_es`	|		Text generation | 	Commonsense reasoning|
|FLORES	|`flores_en-es`	| Sequence-to-sequence generation|	Machine translation|
|INCLUDE|	`include_base_44_spanish`		|	Multiple-choice QA	|Language-specific & world knowledge|
|Global-MMLU|	`global_mmlu_full_es`		|	Multiple-choice QA	|Language-specific & world knowledge|

</details>

### Basque

* Benchmark(s): BasqueBench
* Paper(s): [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage(s): N/A
* Language code(s): `eus_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/basque_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/basque_bench)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   | Task type  | Task category |
|:---|:---|:---|:---|
|	Belebele |	`belebele_eus_Latn`	|	 	Multiple-choice QA	|Reading comprehension|
|	EusExams	|`eus_exams_eu`	|		Multiple-choice QA	|Language-specific & world knowledge|
|	EusProfficiency	|`eus_proficiency`	|	 Multiple-choice QA	|Language-specific & world knowledge|
|	EusReading|	`eus_reading`	|	 Multiple-choice QA|	Reading comprehension|
|	EusTrivia	|`eus_trivia`	|	 Multiple-choice QA|	Language-specific & world knowledge|
|	MGSM-eu	|`mgsm_direct_eu`	|		Generative QA	|Mathematical reasoning|
|	PIQA-eu	|`piqa_eu`	|	Multiple-choice QA|	Commonsense reasoning|
|	NLI (Basque GLUE)	|`qnlieu`	|	Text classification|	Entailment|
|	WNLI	|`wnli_eu`	|	Text classification|	Entailment|
|	XCOPA	|`xcopa_eu`	|	Text cassification	|Commonsense reasoning|
|	XNLI	|`xnli_eu_native`	|	Text classification|	Entailment|
|	xStoryCloze|	`xstorycloze_eu`	|		Multiple-choice QA|	Commonsense reasoning|
|	PAWS-eu|	`paws_eu`	|Text classification|	Paraphrasing|
|	ARC-eu|`arc_eu_easy`	|	 	Multiple-choice QA|	Language-specific & world knowledge|
|	ARC-eu|	`arc_eu_challenge`	|		Multiple-choice QA|	Language-specific & world knowledge|
|	FLORES	|`flores_en-eu`	|	Sequence-to-sequence generation|	Machine translation|
|	INCLUDE	|`include_base_44_basque`	|		Multiple-choice QA|	Language-specific & world knowledge|

</details>

### Galician

* Benchmark: GalicianBench
* Paper: [aclanthology.org/2025.coling-main.699](https://aclanthology.org/2025.coling-main.699)
* Homepage: N/A
* Language code(s): `glg_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/galician_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/galician_bench)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness  | Task type  | Task category |
|:---|:---|:---|:---|
|Belebele|	`belebele_`	| Multiple-choice QA	|Reading comprehension|
|FLORES	|`flores_en-gl`|	Sequence-to-sequence generation|	Machine translation|
|GalCoLA|	`galcola`		|Text classification|	Language knowledge|
|MGSM	|`mgsm_direct_gl`		|Generative QA|	Mathematical reasoning|
|OpenBookQA-gl	| `openbookqa_gl`	|Multiple-choice QA	|Language-specific & world knowledge|
|Parafrases-gl	|`parafrases_gl`		|Text classification	|Paraphrasing|
|PAWS-gl|	`paws_gl`	|Text classification	|Paraphrasing|
|TruthfulQA-gl Generation	|`truthfulqa_gl_gen`|	Generative QA	|Truthfulness|
|TruthfulQA-gl Multiple-choice	|`truthfulqa_gl_mc1`|	Multiple-choice QA	|Truthfulness|
|TruthfulQA-gl Multiple-choice|	`truthfulqa_gl_mc2` |		Multiple-choice QA	|Truthfulness|
|VeritasQA-gl Generation	|`veritasqa_gen_gl`|	Generative QA	|Truthfulness|
|VeritasQA-gl Multiple-choice|`veritasqa_mc1_gl`|	Multiple-choice QA	|Truthfulness|
|VeritasQA-gl Multiple-choice|	`veritasqa_mc2_gl`|	Multiple-choice QA	|Truthfulness|

</details>

### Norwegian

* Benchmark: NorEval
* Paper: [arxiv.org/abs/2504.07749](https://arxiv.org/abs/2504.07749)
* Homepage: [github.com/ltgoslo/noreval](https://github.com/ltgoslo/noreval/tree/main)
* Language code: `nob_Latn`, `nno_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/noreval)


<details >
<summary><b>Tasks</b></summary>

|Name  |Bokmål | Nynorsk  |Task type  | Task category |
|:---|:---|:---|:---|:---|
|[NoReC Sentence](https://huggingface.co/datasets/ltg/norec_sentence) |```norec_sentence```  | ❌ |Text classification| Sentiment analysis |
|[NoReC Document](https://huggingface.co/datasets/ltg/norec_document) |```norec_document```  | ❌ |Text classification| Sentiment analysis |
|[NCB](https://huggingface.co/datasets/hcfa/ncb) |```ncb```| ❌ | Sentence ranking| Language knowledge   |
|[NorIdiom](https://huggingface.co/datasets/Sprakbanken/Norwegian_idioms) |```noridiom_nob```  | ```noridiom_nno```  | Sentence completion| Language knowledge  |
|[Belebele](https://huggingface.co/datasets/facebook/belebele) |```norbelebele```| ❌|Multiple-choice question answering| Machine reading comprehension |
|[NRK-Quiz-QA](https://huggingface.co/datasets/ltg/nrk_quiz_qa) |```nrk_quiz_qa_nob```| ```nrk_quiz_qa_nno```| Multiple-choice question answering| Language-specific & world knowledge |
|[NorOpenBookQA](https://huggingface.co/datasets/ltg/noropenbookqa) |```noropenbookqa_nob```| ```noropenbookqa_nno``` |Multiple-choice question answering| Language-specific & world knowledge |
|[NorCommonsenseQA](https://huggingface.co/datasets/ltg/norcommonsenseqa) |```norcommonsenseqa_nob```| ```norcommonsenseqa_nno``` |Multiple-choice question answering|Commonsense reasoning  |
|[NorTruthfulQA Multiple choice](https://huggingface.co/datasets/ltg/nortruthfulqa_mc) |```nortruthfulqa_mc_nob```| ```nortruthfulqa_mc_nno``` |Multiple-choice question answering |Truthfulness |
|[NorQuAD](https://huggingface.co/datasets/ltg/norquad) |```norquad```| ❌  | Generative question answering |Machine reading comprehension |
|[NorTruthfulQA Generation](https://huggingface.co/datasets/ltg/nortruthfulqa_gen) |```nortruthfulqa_gen_nob```| ```nortruthfulqa_gen_nno``` |  Generative question answering|Truthfulness |
|[Tatoeba (English → Bokmål/Nynorsk)](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | ```tatoeba_eng_nob```| ```tatoeba_eng_nno```  |Sequence-to-sequence generation|Machine translation |

</details>


### Ukrainian

* Benchmark: N/A
* Paper: N/A
* Homepage: N/A
* Language code: `ukr_Cyrl`
* LM Evaluation Harness: [ukrainian](./ukrainian/)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|[Global-MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU)| `global_mmlu_full_uk` | Multiple-choice QA | Language-specific & world knowledge |
|[ZNO](https://huggingface.co/datasets/osyvokon/zno)| `zno` |  Multiple-choice QA | Language-specific & world knowledge |
|[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-base-44)| `include_base_44_ukrainian` |  Multiple-choice QA | Language-specific & world knowledge |
|[TextDetox](https://huggingface.co/datasets/ukr-detect/ukr-toxicity-dataset)| `textdetox_ukr` | Text classification | Toxicity detection | 
|[UA-SQuAD](https://huggingface.co/datasets/HPLT/ua-squad) | `ua_squad` |  Generative QA | Reading comprehension |
|[Belebele](https://huggingface.co/datasets/facebook/belebele) | `belebele_ukr_Cyrl` |  Multiple-choice QA | Reading comprehension |
|[UA-GEC](https://huggingface.co/datasets/HPLT/ua-gec) | `ua_gec` |  Ranking|	Language knowledge|
|[MultiBLiMP](https://huggingface.co/datasets/jumelet/multiblimp) | `ua_blimp` |  Ranking|	Language knowledge|
|[WMT24PP](https://huggingface.co/datasets/google/wmt24pp/) | `wmt24pp_en-uk` |  Sequence-to-sequence generation	| Machine translation|

</details>

### Czech

* Benchmark: BenCzechMark
* Paper: [arxiv.org/abs/2412.17933](https://arxiv.org/abs/2412.17933)
* Homepage: [github.com/DCGM/lm-evaluation-harness](https://github.com/DCGM/lm-evaluation-harness)
* Language code: `ces_Latn`
* LM Evaluation Harness: [github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark](https://github.com/DCGM/lm-evaluation-harness/tree/main/lm_eval/tasks/benczechmark)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
 |	Belebele	|`belebele_ces_Latn`	 |	 Multiple-choice QA  | 	Reading comprehension |
 |	Global-MMLU|	`global_mmlu_full_cs`	 |		Multiple-choice QA  | 	Language-specific & world knowledge |
 |	SQAD3.2	|`benczechmark_cs_sqad32`	 |	Generative QA	| Reading comprehension| 
 |	Umimeto 	|`benczechmark_umimeto_qa`	 |	Multiple-choice QA  | 	Language-specific & world knowledge| 
 |	CERMAT OPEN	|`benczechmark_cermat_qa`	 |	Generative QA	Language knowledge|
 |	CERMAT TF	|`benczechmark_cermat_czech_tf`	 |	Multiple-choice QA  | 	Language knowledge|
 |	CERMAT MC	|`benczechmark_cermat_mc`	 |	Multiple-choice QA  | 	Language knowledge|
 |	Klokan QA |	`benczechmark_klokan_qa`	 |	Multiple-choice QA  | 	Mathematical reasoning|
 |	CERMAT (Math)|	`benczechmark_cermat_czmath_mc`	 |	Multiple-choice QA  | 	Mathematical reasoning|
 |	Umimeto (Math) |	`benczechmark_umimeto_qa`	 |		Multiple-choice QA  | 	Mathematical reasoning|
 |	CTKFacts 	| `benczechmark_ctkfacts_nli`	 |		Text classification | 	Entailment|
 |	Subjectivity 	|`benczechmark_subjectivity`	 |	Text classification | 	Sentiment analysis|
 |	CzechSentiment - Mall	|`benczechmark_sentiment_mall`	 |		Text classification | 	Sentiment analysis|
 |	CzechSentiment - CSFD |	`benczechmark_sentiment_csfd`	 |		Text classification | 	Sentiment analysis|
 |	CzechSentiment - FB |	`benczechmark_sentiment_fb`	 |	Text classification | 	Sentiment analysis|

</details>

### French

* Benchmark: FrenchBench
* Paper: [arxiv.org/abs/2402.00786](https://arxiv.org/abs/2402.00786)
* Homepage: [huggingface.co/croissantllm](https://huggingface.co/croissantllm)
* Language code: `fra_Latn`
* LM Evaluation Harness: [github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/french_bench)

<details >
<summary><b>Tasks</b></summary>

|Name  |LM Evaluation Harness   |Task type  | Task category |
|:---|:---|:---|:---|
|	FQuaD	|`french_bench_fquadv2`	|					Generative QA|	Reading comprehension|
|	French Trivia|	`french_bench_trivia`	|					Generative QA	|Language-specific & world knowledge|
|	French Language Test: Grammar	|`french_bench_grammar`	|					Multiple-choice QA|	Language knowledge|
|	French Language Test: Vocabulary|	`french_bench_vocab`	|					Multiple-choice QA|	Language knowledge|
|	French Language Test: Reading	|`french_bench_reading_comp`	|					Multiple-choice QA|	Reading comprehension|
|	Belebele	|`belebele_fra_Latn`	|					Multiple-choice QA|	Reading comprehension|
|	French NLI|	`french_bench_topic_based_nli`	|					Text classification|	Entailment|
|	WMT14	|`wmt14-en-fr`	|					Sequence-to-sequence generation|	Machine translation|
|	XNLI	|`french_bench_xnli`	|					Text classification	|Entailment|
|	INCLUDE	|`include_base_44_french`	|	 Multiple-choice QA|	Language-specific & world knowledge|
|	Global-MMLU	| `global_mmlu_fr`	|		Multiple-choice QA|	Language-specific & world knowledge|

</details>


## Installation and usage

`# Will be adapted to LUMI`

Install LM Evaluation Harness as described [here](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#install).

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

### Examples

Detailed guidelines on how to use LM Evaluation Harness can be found [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md).

The task names can be found in the **LM Evaluation Harness** column in the language-specific task tables provided above.


<details>
<summary><b>Basic usage</b></summary>

Below is an example of a basic framework usage and must-have arguments. In general, one needs to pass the tasks with the help of the `--tasks` argument:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=my_hf_model_name \
  --tasks global_mmlu_full_uk,include_base_44_ukrainian \
  --output results/ukrainian/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```
</details>

<details>
<summary><b>Czech</b></summary>

**Step 1: Clone the repository**

```bash
git clone https://github.com/DCGM/lm-evaluation-harness.git
cd lm-evaluation-harness
```

**Step 2: Run the evaluation**

Follow (and test) the instructions [here](https://github.com/DCGM/lm-evaluation-harness/tree/main?tab=readme-ov-file#example-usage).

</details>

<details>
<summary><b>Ukrainian</b></summary>

Evaluation on the Ukrainian tasks requires the usage of the `include_path` argument to ensure our tasks are registered in the framework:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=my_hf_model_name \
  --tasks zno,ua_gec,ua_blimp \
  --include_path ./ukrainian/ \
  --output results/ukrainian/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

</details>


<details>
<summary><b>Task groups</b></summary>

An alternative approach to run all tasks of interest at once involves creating a task group. LM Evaluation Harness allows to group tasks as shown below; please find more details [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md#group-configuration).

**Step 1: Create a configuration file**

Create a configuration file containing the name of the group and corresponding tasks and save it in the, e.g., `groups` folder. An example for the Ukrainian ZNO group task can be found [here](ukrainian/zno/zno.yaml).

```bash
group: hplt_french
task:
  - french_bench_fquadv2
  - french_bench_trivia
  - french_bench_grammar
  - french_bench_vocab
  - french_bench_reading_comp
  - belebele_fra_Latn
  - french_bench_topic_based_nli
  - wmt14-en-fr
  - french_bench_xnli
  - include_base_44_french
  - global_mmlu_fr
```

**Step 2: Run the evaluation**

Here, we are specifying the name of our created group as ```tasks``` and pass the `include_path` argument to ensure our group is registered:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=my_hf_model_name \
  --tasks hplt_french \
  --include_path ./groups/ \
  --output results/hplt_french/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

</details>
